import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional, Dict
import wandb
import logging
import timm
from dataset.data import Dataset
from config import Configuration
from torchcam.methods import GradCAM
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def apply_model_diff(global_model, update_dict):
    """Fixed: removed 'self' parameter"""
    for name, param in global_model.named_parameters():
        if name in update_dict and param.requires_grad:
            param.data += update_dict[name].to(param.device)

def compute_model_diff(model_before, model_after):
    diff = {}
    for (name, p_before), (_, p_after) in zip(model_before.named_parameters(), model_after.named_parameters()):
        if p_before.requires_grad:
            diff[name] = p_after.data.clone() - p_before.data.clone()
    return diff

def copy_model(original_model):
    copied_model = copy.deepcopy(original_model)
    copied_model.to(next(original_model.parameters()).device)
    return copied_model

def get_target_layer_for_model(model):
    """Automatically determine appropriate target layer for GradCAM"""
    if hasattr(model, 'blocks'):  # Vision Transformer
        return model.blocks[-1].norm1
    elif hasattr(model, 'layer4'):  # ResNet
        return model.layer4[-1]
    elif hasattr(model, 'features'):  # VGG, DenseNet, etc.
        return model.features[-1]
    else:
        # Fallback: find the last convolutional or normalization layer
        layers = list(model.modules())
        for layer in reversed(layers):
            if isinstance(layer, (nn.Conv2d, nn.LayerNorm, nn.BatchNorm2d)):
                return layer
        raise ValueError("Could not automatically determine target layer for GradCAM")

def extract_param_feature_map(model, dataloader, device, target_layer, num_samples=50):
    """
    Fixed version: Properly extracts GradCAM-based parameter importance signatures
    """
    model.to(device)
    model.eval()
    
    # Ensure model parameters can compute gradients for GradCAM
    for param in model.parameters():
        param.requires_grad = True
    
    try:
        extractor = GradCAM(model, target_layer=target_layer)
    except Exception as e:
        logging.warning(f"Could not create GradCAM extractor: {e}")
        # Return empty signatures if GradCAM fails
        return {name: 0.0 for name, _ in model.named_parameters() if _.requires_grad}
    
    # Collect activation statistics
    activation_stats = []
    sample_count = 0
    
    try:
        with torch.enable_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                if sample_count >= num_samples:
                    break
                    
                images = images.to(device)
                labels = labels.to(device)
                batch_size = images.size(0)
                
                # Forward pass
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                
                # Process each image in the batch
                for i in range(min(batch_size, num_samples - sample_count)):
                    try:
                        # Generate CAM for this specific prediction
                        single_output = outputs[i:i+1]
                        pred_class = preds[i].item()
                        
                        # Get activation map
                        activation_maps = extractor(pred_class, single_output)
                        
                        if activation_maps and len(activation_maps) > 0:
                            cam = activation_maps[0]
                            # Compute statistics from the activation map
                            activation_stats.append({
                                'mean_activation': cam.mean().item(),
                                'max_activation': cam.max().item(),
                                'std_activation': cam.std().item(),
                                'prediction_confidence': torch.softmax(single_output, dim=1)[0, pred_class].item()
                            })
                        
                        sample_count += 1
                        
                    except Exception as e:
                        logging.warning(f"GradCAM failed for sample {i} in batch {batch_idx}: {e}")
                        continue
                
                # Only process one batch for efficiency
                if sample_count > 0:
                    break
                    
    except Exception as e:
        logging.error(f"Error during GradCAM extraction: {e}")
        # Return uniform signatures if extraction completely fails
        return {name: 1.0 for name, _ in model.named_parameters() if _.requires_grad}
    
    # Convert activation statistics to parameter signatures
    if not activation_stats:
        logging.warning("No activation statistics collected, using uniform signatures")
        return {name: 1.0 for name, _ in model.named_parameters() if _.requires_grad}
    
    # Compute overall activation characteristics
    avg_activation = np.mean([stat['mean_activation'] for stat in activation_stats])
    avg_confidence = np.mean([stat['prediction_confidence'] for stat in activation_stats])
    activation_consistency = 1.0 - np.std([stat['mean_activation'] for stat in activation_stats])
    
    # Map to parameter-level signatures
    param_signatures = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Base signature on activation characteristics
            base_score = abs(avg_activation) * avg_confidence
            
            # Adjust based on parameter type and location
            if 'head' in name or 'classifier' in name:
                # Classification layers are more important
                layer_multiplier = 1.5
            elif 'norm' in name:
                # Normalization layers important for stability
                layer_multiplier = 1.2
            elif any(x in name for x in ['bias', 'scale']):
                # Bias/scale parameters less critical
                layer_multiplier = 0.8
            else:
                layer_multiplier = 1.0
            
            # Incorporate consistency
            consistency_factor = max(0.1, activation_consistency)
            
            param_signatures[name] = base_score * layer_multiplier * consistency_factor
    
    return param_signatures

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

def combine_scores(fisher_info, cam_signatures, alpha=0.5):
    """
    Fixed version: Properly combines Fisher information with GradCAM signatures
    """
    combined = {}
    
    # Normalize Fisher information
    norm_fisher = {}
    for k, v in fisher_info.items():
        v_flat = v.view(-1)
        v_max = v_flat.max()
        if v_max > 1e-8:
            norm_fisher[k] = v / v_max
        else:
            norm_fisher[k] = v
    
    # Create GradCAM tensor signatures matching Fisher tensor shapes
    norm_cam = {}
    for k, v in fisher_info.items():
        cam_score = cam_signatures.get(k, 1.0)  # Default to 1.0 if signature missing
        # Normalize CAM signature to [0, 1] range
        normalized_cam_score = max(0.0, min(1.0, cam_score))
        # Create tensor with same shape as Fisher tensor, filled with normalized CAM score
        norm_cam[k] = torch.full_like(v, normalized_cam_score)
    
    # Combine normalized scores
    for name in fisher_info:
        combined[name] = alpha * norm_fisher[name] + (1 - alpha) * norm_cam[name]
    
    return combined

def generate_global_mask(score_dict, top_k: float = 0.2):
    """Generate mask for most important parameters"""
    all_scores = torch.cat([f.view(-1) for f in score_dict.values()])
    threshold = torch.quantile(all_scores, 1 - top_k)
    mask = {}
    for name, tensor in score_dict.items():
        mask[name] = (tensor >= threshold).float()
    return mask

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    use_gradcam: bool = True,
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
                "use_gradcam": use_gradcam,
            },
        )

    device = get_device()
    model = training_params.model.to(device)
    loss_func = training_params.loss_function
    optimizer = training_params.optimizer
    scheduler = training_params.scheduler
    num_epochs = training_params.epochs
    best_acc = 0

    # ---- Step 1: Compute Fisher Information ----
    logging.info("Computing Fisher information...")
    fisher_info = compute_fisher_information(model, train_loader, device, loss_func, num_samples=fisher_samples)
    
    # ---- Step 2: Compute GradCAM signatures (if enabled) ----
    if use_gradcam:
        logging.info("Computing GradCAM signatures...")
        try:
            target_layer = get_target_layer_for_model(model)
            cam_signatures = extract_param_feature_map(model, train_loader, device, target_layer)
            # Combine Fisher and GradCAM
            combined_scores = combine_scores(fisher_info, cam_signatures, alpha=0.7)
            logging.info("Successfully combined Fisher and GradCAM scores")
        except Exception as e:
            logging.warning(f"GradCAM computation failed: {e}. Using Fisher-only scores.")
            combined_scores = fisher_info
    else:
        logging.info("Using Fisher-only scores")
        combined_scores = fisher_info
    
    # ---- Step 3: Generate mask ----
    global_mask = generate_global_mask(combined_scores, top_k=top_k_mask)

    # ---- Step 4: Training loop ----
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

            # Apply mask to gradients
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
            wandb.log({"Epoch": epoch, "Train Loss": train_loss, "Train Accuracy": train_accuracy})
        else:
            logging.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        if val_loader:
            _, _, val_loss, val_accuracy = compute_predictions(model, val_loader, device, loss_func)

            if wandb_log:
                wandb.log({"Epoch": epoch, "Validation Loss": val_loss, "Validation Accuracy": val_accuracy})
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


# ------------------- Running Training -------------------
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    data = Dataset()
    dino = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
    for param in dino.parameters():
        param.requires_grad = False
    dino.head = nn.Linear(384, 100)

    config = Configuration(
        model=dino,
        training_name="fl_centralized_fisher_gradcam_fixed",
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
    )

    train_dataloader, val_dataloader = data.get_dataloaders(config.dataset)

    res_dict = train_model_with_mask(
        training_params=config,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        project_name="fl_centralized_fisher_gradcam_fixed",
        top_k_mask=0.2,
        use_gradcam=True
    )

    print(f"Training completed. Best accuracy: {res_dict['best_accuracy']:.2f}%")