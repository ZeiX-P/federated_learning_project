import logging
import os
from typing import Optional, Dict, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb



import logging
import os
from typing import Optional, Dict, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_function: Optional[nn.Module] = None,
):
    model.eval()
    predictions = []
    labels = []
    total_loss = 0.0
    total_samples = 0

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
    correct = (predictions == labels).sum().item()
    accuracy = 100.0 * correct / len(labels)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    return predictions, labels, avg_loss, accuracy


def train_model(
    *,
    training_params,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    project_name: Optional[str] = None,
    wandb_log: bool = True,
    wandb_save: bool = True,
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

    best_acc = 0
    num_epochs = training_params.epochs

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

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
            _, _, val_loss, val_accuracy = compute_predictions(
                model=model, dataloader=val_loader, device=device, loss_function=loss_func
            )
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

def train_with_global_mask(self, model, train_loader, val_loader, client_id, round_num):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=self.config.lr)

    for epoch in range(self.config.local_epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = self.config.loss_function(outputs, targets)
            loss.backward()

            # Apply global mask to gradients
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in self.global_mask and param.grad is not None:
                        param.grad.mul_(self.global_mask[name])

            optimizer.step()

        # Optional: evaluate on validation set
        val_loss, val_accuracy = self.evaluate_model(model, val_loader)

        # Log training/validation metrics
        wandb.log({
            f"client_{client_id}/val_loss": val_loss,
            f"client_{client_id}/val_accuracy": val_accuracy,
            "epoch": epoch,
            "round": round_num
        })
