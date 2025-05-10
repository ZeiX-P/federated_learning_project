import logging
import os
from typing import Optional, Dict, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb



def compute_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_function: Optional[nn.Module] = None,
):
    model.eval()
    predictions = []
    labels = []
    loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)

            if loss_function is not None:
                loss += loss_function(preds, targets).item()

            _, predicted = torch.max(preds, 1)
            predictions.append(predicted)
            labels.append(targets)

    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    correct = (predictions == labels).sum().item()
    accuracy = 100.0 * correct / len(labels)
    loss = loss / len(labels)

    return predictions, labels, loss, accuracy


def train_model(
    *,
    training_params,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    project_name: Optional[str] = None,
    wandb_log: bool = True,
    wandb_save: bool = True,
) -> dict:

    #assert isinstance(training_params, Configuration)
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    use_wandb = wandb_log or wandb_save
    if use_wandb:
        if project_name is None:
            raise ValueError("project name cannot be None with either wandb_log or wandb_save set to True.")

        wandb.init(
            project=project_name,
            name=training_params.training_name,
            config={
                "epochs": training_params.epochs,
                "batch_size": train_loader.batch_size,
                "learning_rate": training_params.learning_rate,
                "architecture": training_params.model.__class__.__name__,
                "optimizer_class": training_params.optimizer.__name__,
                "loss_function": training_params.loss_function.__class__.__name__
               
            },
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = training_params.model.to(device)
    loss_func = training_params.loss_function

    optimizer = training_params.optimizer


    scheduler = training_params.scheduler

    best_acc = 0
    num_epochs = training_params.epochs

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        wandb.log({"Epoch": epoch}) 
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            print(f"Batch {batch_idx}/{len(train_loader)}")
            inputs, targets = inputs.to(device), targets.to(device)

            preds = model(inputs)
            loss = loss_func(preds, targets)
            
            optimizer.zero_grad
            loss.backward
            optimizer.step

            running_loss += loss.item()
            _, predicted = preds.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            wandb.log({
                    "batch/train_loss": loss.item(),
                    
                    "batch": epoch * len(train_loader) + batch_idx,
                })

        if scheduler is not None:
            scheduler.step

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total

        if wandb_log:
            wandb.log({
                "Epoch": epoch,
                "Train Loss": train_loss,
                "Train Accuracy": train_accuracy,
            })
        else:
            logging.info(f"Epoch: {epoch}: Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")

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
                logging.info(f"Epoch: {epoch}: Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

            if val_accuracy > best_acc:
                best_acc = val_accuracy
                if wandb_save:
                    model_name = f"{training_params.training_name}_best.pth"
                    torch.save(model.state_dict(), model_name)
                    wandb.save(model_name)

    if use_wandb:
        wandb.finish()

    return {"model": model, "best_accuracy": best_acc}
