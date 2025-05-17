import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict
import wandb
import logging
import timm
from config import Configuration
from dataset.data import Dataset

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_predictions(model, dataloader, device, loss_function=None):
    model.eval()
    predictions, labels = [], []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)

            if loss_function:
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


def compute_fisher_information(model, dataloader, device, loss_function, num_samples=100):
    model.eval()
    fisher_info = {name: torch.zeros_like(p, device=device) for name, p in model.named_parameters() if p.requires_grad}

    count = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_info[name] += param.grad.data.pow(2)

        count += 1
        if count >= num_samples:
            break

    for name in fisher_info:
        fisher_info[name] /= count

    return fisher_info


def apply_model_editing(model, fisher_info: Dict[str, torch.Tensor], original_params: Dict[str, torch.Tensor], alpha: float):
    for name, param in model.named_parameters():
        if name in fisher_info and name in original_params:
            with torch.no_grad():
                # Quadratic penalty on deviation from original weights
                penalty = fisher_info[name] * (param - original_params[name])
                param.copy_(param - alpha * penalty)


def train_model_with_fisher_editing(
    training_params,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    project_name: Optional[str] = None,
    wandb_log: bool = True,
    wandb_save: bool = True,
    apply_editing: bool = True,
    editing_alpha: float = 1e-3,
    fisher_samples: int = 100,
):
    device = get_device()
    model = training_params.model.to(device)
    loss_func = training_params.loss_function
    optimizer = training_params.optimizer
    scheduler = training_params.scheduler
    best_acc = 0

    if wandb_log or wandb_save:
        wandb.init(
            project=project_name,
            name=training_params.training_name,
            config={
                "epochs": training_params.epochs,
                "batch_size": train_loader.batch_size,
                "learning_rate": training_params.learning_rate,
                "architecture": model.__class__.__name__,
            },
        )

    # Save original parameters
    original_params = {name: p.clone().detach() for name, p in model.named_parameters()}

    # Standard training
    for epoch in range(training_params.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = loss_func(preds, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            _, predicted = preds.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        if scheduler:
            scheduler.step()

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        if wandb_log:
            wandb.log({
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Train Accuracy": train_acc,
            })

        if val_loader:
            _, _, val_loss, val_acc = compute_predictions(model, val_loader, device, loss_func)
            if wandb_log:
                wandb.log({
                    "Epoch": epoch,
                    "Validation Loss": val_loss,
                    "Validation Accuracy": val_acc,
                })

            if val_acc > best_acc:
                best_acc = val_acc
                if wandb_save:
                    torch.save(model.state_dict(), f"{training_params.training_name}_best.pth")
                    wandb.save(f"{training_params.training_name}_best.pth")

    # Model editing after training
    if apply_editing:
        fisher_info = compute_fisher_information(model, train_loader, device, loss_func, fisher_samples)
        apply_model_editing(model, fisher_info, original_params, alpha=editing_alpha)

    if wandb_log or wandb_save:
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

res_dict = train_model_with_fisher_editing(
    training_params=config,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    project_name="fl_centralized_fisher",
    apply_editing=True,
    editing_alpha=1e-3,
    fisher_samples=100,
)