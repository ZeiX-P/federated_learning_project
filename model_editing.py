
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
import numpy as np

# Assuming dataset.data and config are properly set up
from dataset.data import Dataset
from config import Configuration

import timm

# Configure logging to show informational messages
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def get_device():
    """
    Determines and returns the appropriate device (CUDA if available, otherwise CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_fisher_information(model, dataloader, device, loss_fn, num_samples=100):
    """
    Computes the diagonal Fisher Information Matrix for the model's parameters.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for a subset of the training data.
        device (torch.device): The device to run computations on.
        loss_fn (nn.Module): The loss function used for training.
        num_samples (int): Number of batches to use for Fisher calculation.

    Returns:
        dict: A dictionary where keys are parameter names and values are their
              corresponding Fisher information (squared gradients averaged).
    """
    model.eval()
    fisher = {}
    # Initialize Fisher information for all trainable parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param)

    count = 0
    # Iterate over a subset of the dataloader to estimate Fisher information
    for inputs, targets in dataloader:
        if count >= num_samples:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad() # Zero gradients for this batch
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward() # Compute gradients

        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data ** 2

        count += 1

    # Average the squared gradients over the sampled batches
    if count > 0:
        for name in fisher:
            fisher[name] /= count
    else:
        logging.warning("No samples processed for Fisher Information calculation. Fisher values will be zero.")
        # If no samples, fisher remains all zeros, which might not be desired.
        # Consider raising an error or handling this case based on expected behavior.

    return fisher

def generate_global_mask1(model, top_k: float = 0.2, strategy: str = "fisher_least",
                          fisher_info: Optional[dict] = None, quantile_sample_size: int = 1_000_000):
    """
    Generates a global mask for model parameters based on importance scores.

    Args:
        model (nn.Module): The neural network model.
        top_k (float): Percentage of parameters to keep trainable (0.0 to 1.0).
                       For 'fisher_least'/'magnitude_lowest', this is the % of least important.
                       For 'fisher_most'/'magnitude_highest', this is the % of most important.
        strategy (str): Pruning strategy ("fisher_least", "fisher_most",
                        "magnitude_lowest", "magnitude_highest", "random").
        fisher_info (Optional[dict]): Pre-computed Fisher information, required for "fisher" strategies.
        quantile_sample_size (int): Max number of elements to sample for quantile calculation
                                    to avoid RuntimeError for very large tensors.

    Returns:
        tuple:
            - mask (dict): Dictionary of binary masks (1.0 for trainable, 0.0 for frozen)
                           for each parameter tensor.
            - all_scores_for_logging (torch.Tensor or None): Concatenated (and possibly sampled)
                                                             importance scores for logging.
    """
    all_scores_for_logging = None # Initialize to None

    if strategy.startswith("fisher"):
        if fisher_info is None:
            raise ValueError("fisher_info must be provided for 'fisher' strategies.")
        all_scores_list = [f.view(-1) for f in fisher_info.values()]
    elif strategy.startswith("magnitude"):
        # Collect absolute values of all trainable parameters
        all_scores_list = [p.data.abs().view(-1) for p in model.parameters() if p.requires_grad]
    elif strategy == "random":
        mask = {
            name: (torch.rand_like(param) < top_k).float()
            for name, param in model.named_parameters() if param.requires_grad
        }
        return mask, None # No meaningful scores to log for random strategy
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    total_elements = sum(s.numel() for s in all_scores_list)

    # Sample if the total number of elements for quantile calculation is too large
    if total_elements > quantile_sample_size:
        sampled_scores_list = []
        for s in all_scores_list:
            num_to_sample_per_tensor = int(s.numel() / total_elements * quantile_sample_size)
            if num_to_sample_per_tensor > 0:
                idx = torch.randperm(s.numel(), device=s.device)[:num_to_sample_per_tensor]
                sampled_scores_list.append(s[idx])
        all_scores_for_logging = torch.cat(sampled_scores_list)
        logging.info(f"Quantile: Sampled {all_scores_for_logging.numel()} elements for quantile calculation from {total_elements} total.")
    else:
        all_scores_for_logging = torch.cat(all_scores_list)

    # Determine the threshold based on strategy
    if strategy == "fisher_least" or strategy == "magnitude_lowest":
        # Keep parameters whose scores are <= threshold
        threshold = torch.quantile(all_scores_for_logging, top_k)
        compare = lambda x: x <= threshold
    elif strategy == "fisher_most" or strategy == "magnitude_highest":
        # Keep parameters whose scores are >= threshold
        threshold = torch.quantile(all_scores_for_logging, 1 - top_k)
        compare = lambda x: x >= threshold
    else:
        # This case should ideally be caught earlier, but for safety
        raise ValueError(f"Unexpected strategy during thresholding: {strategy}")

    # Apply the comparison to the original (full) parameter tensors
    mask = {}
    if strategy.startswith("fisher"):
        source_data = fisher_info
    elif strategy.startswith("magnitude"):
        source_data = {name: param.data for name, param in model.named_parameters() if param.requires_grad}

    for name, data_tensor in source_data.items():
        if data_tensor.numel() > 0:
            # For magnitude, apply abs() before comparison
            mask[name] = compare(data_tensor.abs() if strategy.startswith("magnitude") else data_tensor).float()
        else:
            # Handle empty tensors - default to trainable (mask=1)
            mask[name] = torch.ones_like(data_tensor).float()

    return mask, all_scores_for_logging

def compute_predictions(model: nn.Module, dataloader: DataLoader, device: torch.device, loss_function: Optional[nn.Module] = None):
    """
    Computes predictions, loss, and accuracy for a given model and dataloader.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the dataset to evaluate.
        device (torch.device): The device to run computations on.
        loss_function (Optional[nn.Module]): The loss function to compute loss.

    Returns:
        tuple: (predictions, labels, average_loss, accuracy)
    """
    model.eval() # Set model to evaluation mode
    predictions, labels = [], []
    total_loss, total_samples = 0.0, 0

    with torch.no_grad(): # Disable gradient calculations
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)

            if loss_function is not None:
                total_loss += loss_function(preds, targets).item() * targets.size(0)
                total_samples += targets.size(0)

            _, predicted = torch.max(preds, 1) # Get the index of the max log-probability
            predictions.append(predicted.cpu())
            labels.append(targets.cpu())

    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    accuracy = 100.0 * (predictions == labels).sum().item() / len(labels)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    return predictions, labels, avg_loss, accuracy

def train_model_with_mask(
    *, # Enforce keyword-only arguments after this point
    training_params,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    project_name: Optional[str] = None,
    wandb_log: bool = True,
    wandb_save: bool = True,
    fisher_samples: int = 100,
    top_k_mask: float = 0.2, # percentage of parameters to keep trainable (based on strategy)
    strategy: str = "fisher_least", # Strategy for mask generation
    quantile_sample_size: int = 1_000_000,
) -> dict:
    """
    Trains a model after applying a mask to freeze/unfreeze parameters based on importance.

    Args:
        training_params: An object containing model, optimizer_class, loss_function, etc.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (Optional[DataLoader]): DataLoader for validation data.
        project_name (Optional[str]): WandB project name.
        wandb_log (bool): Whether to log metrics to WandB.
        wandb_save (bool): Whether to save best model checkpoints to WandB.
        fisher_samples (int): Number of batches to use for Fisher calculation (if strategy is Fisher-based).
        top_k_mask (float): Percentage of parameters to keep trainable.
        strategy (str): Pruning strategy ("fisher_least", "fisher_most", etc.).
        quantile_sample_size (int): Max number of elements to sample for quantile calculation.

    Returns:
        dict: Contains the trained model and its best validation accuracy.
    """
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
                "pruning_strategy": strategy, # Log the strategy used
                "top_k_mask": top_k_mask,    # Log the top_k_mask
                "fisher_samples": fisher_samples,
                "quantile_sample_size": quantile_sample_size
            },
        )

    device = get_device()
    model = training_params.model.to(device)
    loss_func = training_params.loss_function
    original_optimizer_class = training_params.optimizer_class
    original_optimizer_params = training_params.optimizer_params
    scheduler = training_params.scheduler_class(None, **training_params.scheduler_params) if training_params.scheduler_class else None
    num_epochs = training_params.epochs
    best_acc = 0

    # --- Step 1: Compute Fisher Info if needed, then Generate Mask ---
    fisher_info = None
    if strategy.startswith("fisher"):
        logging.info(f"Computing Fisher Information using {fisher_samples} samples...")
        fisher_info = compute_fisher_information(model, train_loader, device, loss_func, num_samples=fisher_samples)
        logging.info("Fisher Information computed.")
    elif strategy.startswith("magnitude") or strategy == "random":
        logging.info(f"Using '{strategy}' strategy for parameter importance.")
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")


    logging.info(f"Generating global mask with strategy '{strategy}' and top_k_mask={top_k_mask}...")
    global_mask_for_trainable, all_scores_for_logging = generate_global_mask1(
        model=model,
        fisher_info=fisher_info, # Pass fisher_info only if computed
        top_k=top_k_mask,
        strategy=strategy,
        quantile_sample_size=quantile_sample_size
    )
    logging.info("Global mask generated.")


    # --- LOG IMPORTANCE VALUES TO WANDB ---
    if wandb_log and all_scores_for_logging is not None:
        log_prefix = "Magnitude" if strategy.startswith("magnitude") else "Fisher"
        logging.info(f"Logging {log_prefix} information histogram and statistics to WandB.")
        scores_np = all_scores_for_logging.cpu().numpy()

        wandb.log({f"{log_prefix}/Values_Histogram": wandb.Histogram(scores_np)}, step=0)

        num_exact_zeros = (scores_np == 0.0).sum()
        total_elements_in_sample = len(scores_np)

        wandb.log({
            f"{log_prefix}/Min_Value": np.min(scores_np),
            f"{log_prefix}/Max_Value": np.max(scores_np),
            f"{log_prefix}/Mean_Value": np.mean(scores_np),
            f"{log_prefix}/Median_Value": np.median(scores_np),
            f"{log_prefix}/Std_Dev_Value": np.std(scores_np),
            f"{log_prefix}/Exact_Zero_Values_Count": num_exact_zeros,
            f"{log_prefix}/Exact_Zero_Values_Percentage": (num_exact_zeros / total_elements_in_sample * 100) if total_elements_in_sample > 0 else 0
        }, step=0)

        quantiles_to_log = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95, 0.99]
        for q_val in quantiles_to_log:
            try:
                q_threshold = torch.quantile(torch.from_numpy(scores_np), q_val).item()
                wandb.log({f"{log_prefix}/Quantile_{int(q_val*100)}percentile": q_threshold}, step=0)
            except RuntimeError as e:
                logging.warning(f"Could not compute quantile for {q_val*100}%: {e}")
            except Exception as e: # Catch other potential errors, e.g., if scores_np is too small
                logging.warning(f"Error logging quantile {q_val*100}%: {e}")


    # --- Apply `requires_grad_(False)` based on the mask ---
    total_params_count = 0
    frozen_params_count = 0
    for name, param in model.named_parameters():
        total_params_count += param.numel()

        if name in global_mask_for_trainable:
            mask_for_param = global_mask_for_trainable[name]

            # If ALL elements in the mask for this parameter are 0.0, freeze the entire parameter
            if (mask_for_param == 0).all():
                param.requires_grad_(False)
                frozen_params_count += param.numel()
            else: # Otherwise, keep it trainable
                param.requires_grad_(True)
        else:
            # For parameters not explicitly in the mask (e.g., a newly added head,
            # or if fisher_info wasn't computed for all), keep them trainable.
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
        }, step=0) # Log at step 0 as this is a setup metric
    else:
        logging.info(f"Freezing: {frozen_params_count}/{total_params_count} parameters frozen "
                     f"({sparsity_percentage_frozen:.2f}%). "
                     f"Trainable: {trainable_percentage:.2f}%")

    # --- Re-initialize the optimizer with ONLY the trainable parameters ---
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logging.info(f"Initializing optimizer with {len(trainable_params)} trainable parameter groups.")
    optimizer = original_optimizer_class(
        trainable_params,
        lr=training_params.learning_rate,
        **original_optimizer_params
    )
    if scheduler:
        # If scheduler is instantiated with a None optimizer initially, update it
        scheduler.optimizer = optimizer


    # --- Step 2: Train Model (now with permanently frozen parameters) ---
    logging.info(f"Starting training for {num_epochs} epochs.")
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
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
            if wandb_log:
                wandb.log({"Learning Rate": optimizer.param_groups[0]['lr']}, step=epoch)

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
                logging.info(f"New best validation accuracy: {best_acc:.2f}%")
                if wandb_save:
                    model_name = f"{training_params.training_name}_best.pth"
                    torch.save(model.state_dict(), model_name)
                    wandb.save(model_name)
        logging.info(f"Finished epoch {epoch}. Current best accuracy: {best_acc:.2f}%")

    if use_wandb:
        wandb.finish()
    logging.info("Training complete.")

    return {"model": model, "best_accuracy": best_acc}

# --- Setup Model and Configuration ---

data = Dataset()
dino = timm.create_model('vit_small_patch16_224.dino', pretrained=True)

# Ensure all parameters are trainable initially, including the backbone.
# This makes sure our freezing mechanism is the one deciding.
for param in dino.parameters():
    param.requires_grad = True

# Replace the head - parameters of the new head are automatically requires_grad=True
dino.head = nn.Linear(384, 100) # Output 100 classes for CIFAR100

# Example Configuration (ensure your config.py defines the Configuration class properly)
# config.py content might look like:
# from dataclasses import dataclass
# from torch.nn import Module
# from torch.optim import Optimizer
#
# @dataclass
# class Configuration:
#     model: Module
#     training_name: str
#     batch_size: int
#     learning_rate: float
#     momentum: float
#     weight_decay: float
#     dataset: str
#     optimizer_class: type[Optimizer]
#     loss_function: Module
#     epochs: int
#     optimizer_params: dict
#     scheduler_class: Optional[type] = None
#     scheduler_params: Optional[dict] = None
#     project_name: Optional[str] = None


config = Configuration(
    model=dino,
    training_name="fl_centralized_baseline",
    batch_size=64,
    learning_rate=1e-3,
    momentum=0.9, # Example: momentum for SGD
    weight_decay=5e-4, # Example: weight decay for SGD
    dataset="CIFAR100",
    optimizer_class=torch.optim.SGD, # Using SGD
    loss_function=nn.CrossEntropyLoss(), # Using CrossEntropyLoss for classification
    scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR, # Using CosineAnnealingLR
    epochs=25,
    optimizer_params={"momentum": 0.9, "weight_decay": 5e-4}, # Params specific to SGD
    scheduler_params={"T_max": 20}, # Params specific to CosineAnnealingLR
    project_name="fl_centralized_model_editing",
)

train_dataloader, val_dataloader = data.get_dataloaders(config.dataset, batch_size=config.batch_size) # Pass batch_size


# --- Call with Magnitude Pruning (Example) ---
logging.info("\n--- Starting run with Magnitude Pruning (Lowest) ---")
res_dict_magnitude = train_model_with_mask(
    training_params=config,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    project_name="fl_centralized_magnitude_editing",
    top_k_mask=0.2, # Keep the 20% parameters with LEAST magnitude trainable.
                    # This means 80% with highest magnitude are frozen.
    strategy="magnitude_lowest",
    quantile_sample_size=10_000_000
)
logging.info(f"Magnitude Pruning Run Best Accuracy: {res_dict_magnitude['best_accuracy']:.2f}%")


# --- Call with Fisher Pruning (Example) ---
# It's good practice to re-initialize the model or reload its state if you're running
# multiple experiments sequentially that modify the model in-place (like freezing).
# For simplicity, here I'll just re-create the DINO model and config for the second run.
logging.info("\n--- Starting run with Fisher Pruning (Most) ---")
dino_fisher = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
for param in dino_fisher.parameters():
    param.requires_grad = True
dino_fisher.head = nn.Linear(384, 100)

config_fisher = Configuration(
    model=dino_fisher, # Use the new model instance
    training_name="fl_centralized_fisher_pruning",
    batch_size=config.batch_size,
    learning_rate=config.learning_rate,
    momentum=config.momentum,
    weight_decay=config.weight_decay,
    dataset=config.dataset,
    optimizer_class=config.optimizer_class,
    loss_function=config.loss_function,
    scheduler_class=config.scheduler_class,
    epochs=config.epochs,
    optimizer_params=config.optimizer_params,
    scheduler_params=config.scheduler_params,
    project_name="fl_centralized_model_editing", # Can use same project or different
)

res_dict_fisher = train_model_with_mask(
    training_params=config_fisher, # Use the new config instance
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    project_name="fl_centralized_fisher_editing",
    top_k_mask=0.2, # Keep the 20% MOST important parameters trainable.
                    # This means 80% with least Fisher are frozen.
    strategy="fisher_most",
    quantile_sample_size=10_000_000
)
logging.info(f"Fisher Pruning Run Best Accuracy: {res_dict_fisher['best_accuracy']:.2f}%")