import random
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional
import wandb
import logging
import timm
from dataset.data import Dataset
from torch.utils.data import Subset
from config import Configuration
from collections import defaultdict

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
        
        # Use kthvalue for better memory efficiency with large tensors
        total_elements = all_scores.numel()
        
        if strategy == "fisher_least":
            k = max(1, int(top_k * total_elements))
            threshold = torch.kthvalue(all_scores, k).values
            compare = lambda x: x <= threshold
        elif strategy == "fisher_most":
            k = max(1, int((1 - top_k) * total_elements))
            threshold = torch.kthvalue(all_scores, k).values
            compare = lambda x: x >= threshold
        elif strategy == "fisher_left_only":
            # New strategy: only parameters on the left side of distribution (least important)
            # This sets mask to 1 ONLY for the leftmost top_k fraction of Fisher values
            k = max(1, int(top_k * total_elements))
            threshold = torch.kthvalue(all_scores, k).values
            compare = lambda x: x <= threshold
        else:
            raise ValueError(f"Unknown Fisher strategy: {strategy}")
        
        mask = {name: compare(tensor).float() for name, tensor in fisher_info.items()}

    elif strategy in {"magnitude_lowest", "magnitude_highest"}:
        all_params = torch.cat([p.view(-1).abs() for p in fisher_info.values()])
        total_elements = all_params.numel()
        
        if strategy == "magnitude_lowest":
            k = max(1, int(top_k * total_elements))
            threshold = torch.kthvalue(all_params, k).values
            compare = lambda x: x.abs() <= threshold
        else:
            k = max(1, int((1 - top_k) * total_elements))
            threshold = torch.kthvalue(all_params, k).values
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
    total_elements = all_scores.numel()
    k = max(1, int((1 - top_k) * total_elements))
    threshold = torch.kthvalue(all_scores, k).values

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

def idd_split(dataset: Dataset, num_clients: int):
    indices_clients = {i: [] for i in range(num_clients)}
    indices = list(range(len(dataset)))
    data_per_client = len(dataset) // num_clients

    np.random.shuffle(indices)

    for i in range(num_clients):
        start_idx = i * data_per_client
        end_idx = (i + 1) * data_per_client
        client_indices = indices[start_idx:end_idx]
        indices_clients[i] = client_indices

    if end_idx < len(dataset):
        remaining_indices = indices[end_idx:]
        for i in range(len(remaining_indices)):
            indices_clients[random.randrange(num_clients)].append(remaining_indices[i])

    return indices_clients

def federated_averaging(
    *,
    dataset,
    num_clients: int,
    global_model: torch.nn.Module,
    training_params,
    iid: bool = True,
    val_split: float = 0.1,
    num_rounds: int = 5,
    batch_size: int = 64,
    project_name: Optional[str] = None,
    wandb_log: bool = True,
    wandb_save: bool = True,
):
    device = get_device()
    global_model = global_model.to(device)

    if wandb_log:
        if project_name is None:
            raise ValueError("project_name must be provided when using wandb.")
        wandb.init(project=project_name, name=training_params.training_name, config={
            "rounds": num_rounds,
            "clients": num_clients,
            "iid": iid,
            "method": "FedAvg"
        })

    if iid:
        indices_map = idd_split(dataset, num_clients)
    else:
        label_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            label_to_indices[label].append(idx)
        for label in label_to_indices:
            random.shuffle(label_to_indices[label])
        shards = []
        for label, idxs in label_to_indices.items():
            shards.extend([idxs[i:i+10] for i in range(0, len(idxs), 10)])
        random.shuffle(shards)
        indices_map = {i: [] for i in range(num_clients)}
        for i, shard in enumerate(shards):
            indices_map[i % num_clients].extend(shard)

    client_loaders = []
    for client_id in range(num_clients):
        client_indices = indices_map[client_id]
        client_subset = Subset(dataset, client_indices)
        val_size = int(len(client_subset) * val_split)
        train_size = len(client_subset) - val_size
        train_data, val_data = torch.utils.data.random_split(client_subset, [train_size, val_size])
        client_loaders.append((
            DataLoader(train_data, batch_size=batch_size, shuffle=True),
            DataLoader(val_data, batch_size=batch_size)
        ))

    best_global_acc = 0
    for rnd in range(1, num_rounds + 1):
        client_models = []
        client_accuracies = []

        for train_loader, val_loader in client_loaders:
            local_model = copy.deepcopy(global_model).to(device)
            local_model.train()

            optimizer = training_params.optimizer_class(
                local_model.parameters(),
                lr=training_params.learning_rate,
                **training_params.optimizer_params
            )

            for epoch in range(training_params.local_epochs):
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = local_model(inputs)
                    loss = training_params.loss_function(outputs, targets)
                    loss.backward()
                    optimizer.step()

            client_models.append(copy.deepcopy(local_model.state_dict()))
            _, _, _, acc = compute_predictions(local_model, val_loader, device, training_params.loss_function)
            client_accuracies.append(acc)

        new_state_dict = copy.deepcopy(global_model.state_dict())
        for key in new_state_dict:
            new_state_dict[key] = torch.stack([cm[key] for cm in client_models], dim=0).mean(dim=0)
        global_model.load_state_dict(new_state_dict)

        avg_client_acc = sum(client_accuracies) / num_clients
        if wandb_log:
            wandb.log({"Round": rnd, "Avg Client Accuracy": avg_client_acc})

        if avg_client_acc > best_global_acc:
            best_global_acc = avg_client_acc
            if wandb_save:
                model_name = f"{training_params.training_name}_fedavg_best.pth"
                torch.save(global_model.state_dict(), model_name)
                wandb.save(model_name)

    if wandb_log:
        wandb.finish()

    return {"model": global_model, "best_accuracy": best_global_acc}

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
    mask_strategy: str = "fisher_least",
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
                "mask_strategy": mask_strategy,
                "top_k_mask": top_k_mask,
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
    global_mask = generate_global_mask1(fisher_info, top_k=top_k_mask, strategy=mask_strategy)

    # ---- Log how many parameters were masked ----
    total_params = 0
    zeroed_params = 0
    for name, mask in global_mask.items():
        total_params += mask.numel()
        zeroed_params += (mask == 0).sum().item()

    if wandb_log:
        wandb.log({
            "Masking/Total Parameters": total_params,
            "Masking/Zeroed Parameters": zeroed_params,
            "Masking/Sparsity (%)": 100.0 * zeroed_params / total_params,
            "Masking/Strategy": mask_strategy
        })
    else:
        logging.info(f"Masking: {zeroed_params}/{total_params} parameters set to 0 "
                     f"({100.0 * zeroed_params / total_params:.2f}%) using strategy '{mask_strategy}'")

    # --- Apply `requires_grad_(False)` based on the mask ---
    # Parameters with a 0 in the mask will be frozen
    # Parameters with a 1 in the mask will remain trainable
    for name, param in model.named_parameters():
        if name in global_mask:
            
            if (global_mask[name] == 0).all(): # This assumes global_mask is binary after generate_global_mask1
                param.requires_grad_(False)
            else:
                # If any part of the mask is 1, ensure it's trainable (it might have been frozen before)
                param.requires_grad_(True)
        
    optimizer_params = [p for p in model.parameters() if p.requires_grad]
  
    training_params.optimizer = training_params.optimizer_class(
        optimizer_params,
        lr=training_params.learning_rate,
        **training_params.optimizer_params
    )
    optimizer = training_params.optimizer # Update the local optimizer variable

    # ---- Step 2: Train Model with Mask (now applied via requires_grad) ----
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

            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            _, predicted = preds.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total

        if scheduler is not None:
            scheduler.step()

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
    param.requires_grad = False # Ensure all parameters are trainable initially

dino.head = nn.Linear(384, 100) 

config = Configuration(
    model=dino,
    training_name="fl_centralized_editing",
    batch_size=64,
    learning_rate=0.001,
    momentum=0.9,
    weight_decay=5e-4,
    dataset="CIFAR100",
    optimizer_class=torch.optim.SGD,
    loss_function=nn.CrossEntropyLoss(),
    scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
    epochs=15,
    optimizer_params={"momentum": 0.9, "weight_decay": 5e-4},
    scheduler_params={"T_max": 20},
    project_name="fl_centralized_model_editing",
)

train_dataloader, val_dataloader = data.get_dataloaders(config.dataset)

'''
federated_averaging(
    dataset=data.get_dataset(config.dataset, apply_transform=True),
    num_clients=10,
    global_model=dino,
    training_params=config,
    iid=True,  # Set to True for IID data distribution
    num_rounds=15,
    batch_size=config.batch_size,
    project_name=config.project_name,
    wandb_log=True,
    wandb_save=True
)

'''
res_dict = train_model_with_mask(
    training_params=config,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    project_name="fl_centralized_fisher_editing",
    top_k_mask=0.1,  # keep top 20% least important parameters (freeze bottom 80%)
    mask_strategy="fisher_left_only"  # Use the new strategy
)
