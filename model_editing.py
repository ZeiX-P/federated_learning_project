
'''
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
    global_mask = generate_global_mask1(fisher_info, top_k=top_k_mask, strategy="fisher_least")

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
            "Masking/Sparsity (%)": 100.0 * zeroed_params / total_params
        })
    else:
        logging.info(f"Masking: {zeroed_params}/{total_params} parameters set to 0 "
                     f"({100.0 * zeroed_params / total_params:.2f}%)")

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

# --- Setup Model and Configuration ---

data = Dataset()
dino = timm.create_model('vit_small_patch16_224.dino', pretrained=True)

#for param in dino.parameters():
#    param.requires_grad = False
dino.head = nn.Linear(384, 100)

config = Configuration(
    model=dino,
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
    scheduler_params={"T_max": 20},
    project_name="fl_centralized_model_editing",
)

train_dataloader, val_dataloader = data.get_dataloaders(config.dataset)

res_dict = train_model_with_mask(
    training_params=config,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    project_name="fl_centralized_fisher_editing",
    top_k_mask=0.2  # keep top 20% most important parameters
)
'''

import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional
import wandb
import logging
import timm
import numpy as np # Import numpy for histogram binning if needed
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

def generate_global_mask1(fisher_info, top_k: float = 0.2, strategy: str = "fisher_least", quantile_sample_size: int = 1_000_000):
    """
    Generates a global mask based on Fisher information or magnitude.
    top_k: percentage of parameters to keep (for 'fisher_least', this is the percentage of least important).
    strategy: "fisher_least", "fisher_most", "magnitude_lowest", "magnitude_highest", "random"
    quantile_sample_size: Max number of elements to sample for quantile calculation to avoid RuntimeErrors.
    """
    if strategy.startswith("fisher"):
        # Gather all scores into a single tensor, but sample if too large
        all_scores_list = [f.view(-1) for f in fisher_info.values()]

        # Check total number of elements
        total_elements = sum(f.numel() for f in fisher_info.values())

        if total_elements > quantile_sample_size:
            # Sample elements if the total is too large
            sampled_scores_list = []
            for f in all_scores_list:
                num_to_sample_per_tensor = int(f.numel() / total_elements * quantile_sample_size)
                if num_to_sample_per_tensor > 0:
                    idx = torch.randperm(f.numel(), device=f.device)[:num_to_sample_per_tensor] # ensure idx is on same device
                    sampled_scores_list.append(f[idx])
            all_scores = torch.cat(sampled_scores_list)
            logging.info(f"Quantile: Sampled {all_scores.numel()} elements for quantile calculation from {total_elements} total.")
        else:
            all_scores = torch.cat(all_scores_list)

        if strategy == "fisher_least":
            # If we want to KEEP the top_k (e.g., 20%) LEAST important,
            # then the threshold means anything BELOW or EQUAL to it is kept (mask=1),
            # and anything ABOVE it is frozen (mask=0).
            threshold = torch.quantile(all_scores, top_k)
            # 'compare' now returns True for parameters to be KEPT (mask=1)
            compare = lambda x: x <= threshold
        elif strategy == "fisher_most":
            threshold = torch.quantile(all_scores, 1 - top_k)
            compare = lambda x: x >= threshold
        else:
            raise ValueError(f"Unknown Fisher strategy: {strategy}")
        mask = {name: compare(tensor).float() for name, tensor in fisher_info.items()}

    elif strategy in {"magnitude_lowest", "magnitude_highest"}:
        # Similar sampling logic for magnitude if needed
        all_params_list = [p.view(-1).abs() for p in fisher_info.values()]
        total_elements = sum(p.numel() for p in fisher_info.values())

        if total_elements > quantile_sample_size:
            sampled_params_list = []
            for p in all_params_list:
                num_to_sample_per_tensor = int(p.numel() / total_elements * quantile_sample_size)
                if num_to_sample_per_tensor > 0:
                    idx = torch.randperm(p.numel(), device=p.device)[:num_to_sample_per_tensor] # ensure idx is on same device
                    sampled_params_list.append(p[idx])
            all_params = torch.cat(sampled_params_list)
            logging.info(f"Quantile: Sampled {all_params.numel()} elements for magnitude quantile calculation from {total_elements} total.")
        else:
            all_params = torch.cat(all_params_list)

        if strategy == "magnitude_lowest":
            threshold = torch.quantile(all_params, top_k)
            compare = lambda x: x.abs() <= threshold
        else:
            threshold = torch.quantile(all_params, 1 - top_k)
            compare = lambda x: x.abs() >= threshold
        mask = {name: compare(p).float() for name, p in fisher_info.items()}

    elif strategy == "random":
        mask = {
            name: (torch.rand_like(p) < top_k).float() # top_k percentage of parameters are kept (mask=1)
            for name, p in fisher_info.items()
        }

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Return all_scores for logging, even if it's a sampled version
    return mask, all_scores if 'all_scores' in locals() else None


def generate_global_mask(fisher_info, top_k: float = 0.2):
    # This function is not used in the main training flow, but kept for completeness
    all_scores = torch.cat([f.view(-1) for f in fisher_info.values()])
    threshold = torch.quantile(all_scores, 1 - top_k)

    mask = {}
    for name, tensor in fisher_info.items():
        mask[name] = (tensor >= threshold).float() # Keep these (mask = 1)
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
    top_k_mask: float = 0.2, # keep top 20% least important parameters
    quantile_sample_size: int = 1_000_000, # Added this parameter
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
    original_optimizer_class = training_params.optimizer_class
    original_optimizer_params = training_params.optimizer_params
    optimizer = None # Placeholder, will be initialized later
    scheduler = training_params.scheduler
    num_epochs = training_params.epochs
    best_acc = 0

    # --- Step 1: Compute Fisher Info and Generate Mask ---
    fisher_info = compute_fisher_information(model, train_loader, device, loss_func, num_samples=fisher_samples)
    global_mask_for_trainable, all_fisher_scores_sampled = generate_global_mask1(
        fisher_info,
        top_k=top_k_mask,
        strategy="fisher_least", # Or "fisher_most" depending on your goal
        quantile_sample_size=quantile_sample_size
    )

    # --- LOG FISHER VALUES TO WANDB ---
    if wandb_log and all_fisher_scores_sampled is not None:
        logging.info("Logging Fisher information histogram to WandB.")
        # Ensure scores are on CPU and convert to numpy for wandb.Histogram
        fisher_scores_np = all_fisher_scores_sampled.cpu().numpy()

        # You might want to filter out extreme outliers for better visualization
        # For example, filter out values above a certain percentile or focus on non-zero values
        # If there's a huge spike at 0, you might want to visualize with a log scale
        
        # Option 1: Basic histogram
        wandb.log({"Fisher/Fisher_Values_Histogram": wandb.Histogram(fisher_scores_np)}, step=0)

        # Option 2: Log statistics (min, max, median, etc.)
        wandb.log({
            "Fisher/Min_Fisher_Value": np.min(fisher_scores_np),
            "Fisher/Max_Fisher_Value": np.max(fisher_scores_np),
            "Fisher/Mean_Fisher_Value": np.mean(fisher_scores_np),
            "Fisher/Median_Fisher_Value": np.median(fisher_scores_np),
            "Fisher/Std_Fisher_Value": np.std(fisher_scores_np),
            "Fisher/Zero_Values_Count": (fisher_scores_np == 0).sum(),
            "Fisher/Zero_Values_Percentage": (fisher_scores_np == 0).sum() / len(fisher_scores_np) * 100
        }, step=0)

        # Option 3: Log quantiles for specific points (useful for verifying thresholds)
        quantiles_to_log = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99]
        for q_val in quantiles_to_log:
            try:
                q_threshold = torch.quantile(torch.from_numpy(fisher_scores_np), q_val).item()
                wandb.log({f"Fisher/Quantile_{int(q_val*100)}percentile": q_threshold}, step=0)
            except RuntimeError as e:
                logging.warning(f"Could not compute quantile for {q_val*100}%: {e}")

    # --- Apply `requires_grad_(False)` based on the mask ---
    total_params_count = 0
    frozen_params_count = 0
    for name, param in model.named_parameters():
        total_params_count += param.numel()

        if name in global_mask_for_trainable:
            mask_for_param = global_mask_for_trainable[name]

            # If ALL elements in the mask for this parameter are 0.0 (meaning it's among the 80% most important)
            if (mask_for_param == 0).all():
                param.requires_grad_(False)
                frozen_params_count += param.numel()
            else:
                # If any part of the mask is 1.0, it means this parameter IS among the 20% least important
                # and should remain trainable.
                param.requires_grad_(True)
        else:
            # If a parameter is NOT in global_mask_for_trainable (e.g., a new head)
            # ensure it remains trainable.
            param.requires_grad_(True)


    # --- Log how many parameters were frozen ---
    sparsity_percentage_frozen = 100.0 * frozen_params_count / total_params_count
    trainable_percentage = 100.0 - sparsity_percentage_frozen
    if wandb_log:
        wandb.log({
            "Trainable Parameters/Total Parameters": total_params_count,
            "Trainable Parameters/Frozen Parameters": frozen_params_count,
            "Trainable Parameters/Frozen %": sparsity_percentage_frozen,
            "Trainable Parameters/Trainable %": trainable_percentage
        })
    else:
        logging.info(f"Freezing: {frozen_params_count}/{total_params_count} parameters frozen "
                     f"({sparsity_percentage_frozen:.2f}%). "
                     f"Trainable: {trainable_percentage:.2f}%")

    # --- Re-initialize the optimizer with ONLY the trainable parameters ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = original_optimizer_class(
        trainable_params,
        lr=training_params.learning_rate,
        **original_optimizer_params
    )
    if training_params.scheduler_class:
        scheduler = training_params.scheduler_class(optimizer, **training_params.scheduler_params)


    # --- Step 2: Train Model (now with permanently frozen parameters) ---
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

# --- Setup Model and Configuration ---

data = Dataset()
dino = timm.create_model('vit_small_patch16_224.dino', pretrained=True)

for param in dino.parameters():
    param.requires_grad = True

dino.head = nn.Linear(384, 100)

config = Configuration(
    model=dino,
    training_name="fl_centralized_baseline",
    batch_size=64,
    learning_rate=1e-3,
    momentum=0.9,
    weight_decay=5e-4,
    dataset="CIFAR100",
    optimizer_class=torch.optim.SGD,
    loss_function=nn.CrossEntropyLoss(),
    scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
    epochs=10,
    optimizer_params={"momentum": 0.9, "weight_decay": 5e-4},
    scheduler_params={"T_max": 20},
    project_name="fl_centralized_model_editing",
)

train_dataloader, val_dataloader = data.get_dataloaders(config.dataset)

res_dict = train_model_with_mask(
    training_params=config,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    project_name="fl_centralized_fisher_editing",
    top_k_mask=0.1,
    strategy="fisher_least", # Keep the 20% least important parameters trainable.
                             # Consequently, 80% (the most important) will be frozen.
    quantile_sample_size=100000
)