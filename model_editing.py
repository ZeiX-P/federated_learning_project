import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional
import wandb
import logging
import timm 
from dataset.data import Dataset
from config import Configuration

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_fisher_information(model, dataloader, device, loss_fn, num_samples=100):
    model.eval()
    fisher = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param)

    count = 0
    for inputs, targets in dataloader:
        if count >= num_samples:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data ** 2

        count += 1

    for name in fisher:
        fisher[name] /= count

    return fisher

def generate_global_mask1(fisher_info, top_k: float = 0.2, strategy: str = "fisher_least"):
    if strategy.startswith("fisher"):
        all_scores = torch.cat([f.view(-1) for f in fisher_info.values()])
        if strategy == "fisher_least":
            threshold = torch.quantile(all_scores, top_k)
            compare = lambda x: x <= threshold
        elif strategy == "fisher_most":
            threshold = torch.quantile(all_scores, 1 - top_k)
            compare = lambda x: x >= threshold
        else:
            raise ValueError(f"Unknown Fisher strategy: {strategy}")
        mask = {name: compare(tensor).float() for name, tensor in fisher_info.items()}

    elif strategy in {"magnitude_lowest", "magnitude_highest"}:
        all_params = torch.cat([p.view(-1).abs() for p in fisher_info.values()])
        if strategy == "magnitude_lowest":
            threshold = torch.quantile(all_params, top_k)
            compare = lambda x: x.abs() <= threshold
        else:
            threshold = torch.quantile(all_params, 1 - top_k)
            compare = lambda x: x.abs() >= threshold
        mask = {name: compare(p).float() for name, p in fisher_info.items()}

    elif strategy == "random":
        mask = {
            name: (torch.rand_like(p) < top_k).float()
            for name, p in fisher_info.items()
        }

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return mask

def generate_global_mask(fisher_info, top_k: float = 0.2):
    all_scores = torch.cat([f.view(-1) for f in fisher_info.values()])
    threshold = torch.quantile(all_scores, 1 - top_k)

    mask = {}
    for name, tensor in fisher_info.items():
        mask[name] = (tensor >= threshold).float()

    return mask


def compute_predictions(model: nn.Module, dataloader: DataLoader, device: torch.device, loss_function: Optional[nn.Module] = None):
    model.eval()
    predictions, labels = [], []
    total_loss, total_samples = 0.0, 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)

            if loss_function is not None:
                total_loss += loss_function(preds, targets).item() * targets.size(0)
                total_samples += targets.size(0)

            _, predicted = torch.max(preds, 1)
            predictions.append(predicted)
            labels.append(targets)

    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    accuracy = 100.0 * (predictions == labels).sum().item() / len(labels)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    return predictions, labels, avg_loss, accuracy


def train_model_with_mask(
    *,
    training_params,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    project_name: Optional[str] = None,
    wandb_log: bool = True,
    wandb_save: bool = True,
    fisher_samples: int = 100,
    top_k_mask: float = 0.2,
) -> dict:
    assert train_loader is not None
    if val_loader is not None:
        assert isinstance(val_loader, DataLoader)

    use_wandb = wandb_log or wandb_save
    if use_wandb:
        if project_name is None:
            raise ValueError("project_name cannot be None if using wandb.")
        wandb.init(
            project=project_name,
            name=training_params.training_name,
            config={
                "epochs": training_params.epochs,
                "batch_size": train_loader.batch_size,
                "learning_rate": training_params.learning_rate,
                "architecture": training_params.model.__class__.__name__,
            },
        )

    device = get_device()
    model = training_params.model.to(device)
    loss_func = training_params.loss_function
    optimizer = training_params.optimizer
    scheduler = training_params.scheduler
    num_epochs = training_params.epochs
    best_acc = 0

    # ---- Step 1: Compute Fisher Info and Mask ----
    fisher_info = compute_fisher_information(model, train_loader, device, loss_func, num_samples=fisher_samples)
    global_mask = generate_global_mask(fisher_info, top_k=top_k_mask)

    # ---- Step 2: Train Model with Mask ----
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            preds = model(inputs)
            loss = loss_func(preds, targets)

            optimizer.zero_grad()
            loss.backward()

            # Apply Fisher mask to gradients
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in global_mask and param.grad is not None:
                        param.grad.mul_(global_mask[name])

            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            _, predicted = preds.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total

        if wandb_log:
            wandb.log({
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Train Accuracy": train_accuracy,
            })
        else:
            logging.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        if val_loader:
            _, _, val_loss, val_accuracy = compute_predictions(model, val_loader, device, loss_func)

            if wandb_log:
                wandb.log({
                    "Epoch": epoch,
                    "Validation Loss": val_loss,
                    "Validation Accuracy": val_accuracy,
                })
            else:
                logging.info(f"Epoch {epoch}: Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            if val_accuracy > best_acc:
                best_acc = val_accuracy
                if wandb_save:
                    model_name = f"{training_params.training_name}_best.pth"
                    torch.save(model.state_dict(), model_name)
                    wandb.save(model_name)

    if use_wandb:
        wandb.finish()

    return {"model": model, "best_accuracy": best_acc}


data = Dataset()
dino = timm.create_model('vit_small_patch16_224.dino', pretrained=True)


for param in dino.parameters():
    param.requires_grad = False
dino.head = nn.Linear(384, 100)
config = Configuration(
                        model = dino,
                        training_name="fl_centralized_baseline",
                        batch_size=64,
                        learning_rate=1e-3,
                        momentum=0.9,
                        weight_decay=5e-4,
                        dataset="CIFAR100",
                        optimizer_class=torch.optim.SGD,
                        loss_function=nn.CrossEntropyLoss(),
                        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
                        epochs=25,
                        optimizer_params={"momentum": 0.9, "weight_decay": 5e-4},
                        scheduler_params={"T_max": 20})
    
    

train_dataloader, val_dataloader = data.get_dataloaders(config.dataset)

res_dict = train_model_with_mask(
    training_params=config,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    project_name="fl_centralized_fisher_editing",
    top_k_mask=0.2  # keep top 20% most important parameters
)