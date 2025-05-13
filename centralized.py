import torch
import torch.nn as nn
import timm


import logging
import os
from typing import Optional, Dict, Union

import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb



__all__ = ["train_model", "compute_predictions", "train_on_subset"]


os.environ["WANDB_MODE"] = "online"

from typing import List, Tuple, Optional

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split

import attr
import torch
from torch import nn
from typing import Optional, Dict, Any
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def is_nn_module(instance, attribute, value):
    """Validator to check if the value is an instance of nn.Module or its subclass."""
    if not isinstance(value, nn.Module):
        raise TypeError(
            f"{attribute.name} must be an instance of nn.Module or its subclass."
        )
    return value


def is_optimizer_class(instance, attribute, value):
    """Validator to check if the value is a subclass of torch.optim.Optimizer."""
    if not issubclass(value, Optimizer):
        raise TypeError(
            f"{attribute.name} must be a subclass of torch.optim.Optimizer."
        )
    return value


def is_scheduler_class(instance, attribute, value):
    """Validator to check if the value is a subclass of torch.optim.lr_scheduler.LRScheduler"""
    if value is not None:
        if not issubclass(value, LRScheduler):
            raise TypeError(
                f"{attribute.name} must be a subclass of torch.optim.lr_scheduler.LRScheduler"
            )
    return value


@attr.s(frozen=True, kw_only=True)
class TrainingParams:
    """
    A class to store the parameters required for core a model.

    Attributes:
        training_name (str): A name for the core experiment.
        epochs (int): The number of epochs for core.
        learning_rate (float): The learning rate for the optimizer.
        model (nn.Module): The model to be trained.
        optimizer_class (torch.optim.Optimizer): The class of the optimizer to be used for core.
        loss_function (nn.Module): The loss function to be used.
        optimizer_params (Optional[Dict[str, Any]]): A dictionary of additional optimizer parameters (optional).
    """

    training_name: str = attr.ib(validator=attr.validators.instance_of(str))
    epochs: int = attr.ib(validator=attr.validators.instance_of(int))
    learning_rate: float = attr.ib(validator=attr.validators.ge(0.0))
    model: nn.Module = attr.ib(
        validator=is_nn_module
    )  # Custom validation to pass instance check (instance_of checks for exact type and not for superclasses)
    loss_function: nn.Module = attr.ib(validator=is_nn_module)  # Custom validation
    optimizer_class: torch.optim.Optimizer = attr.ib(validator=is_optimizer_class)
    scheduler_class: Optional[torch.optim.lr_scheduler.LRScheduler] = attr.ib(
        validator=is_scheduler_class, default=None
    )
    optimizer_params: Optional[Dict[str, Any]] = attr.ib(default=None)
    scheduler_params: Optional[Dict[str, Any]] = attr.ib(default=None)

    

    @property
    def optimizer(self):
        optimizer_params = self.optimizer_params or {}
        return self.optimizer_class(
            self.model.parameters(), lr=self.learning_rate, **optimizer_params
        )

    @property
    def scheduler(self):
        scheduler_params = self.scheduler_params or {}
        if self.scheduler_class:
            return self.scheduler_class(
                self.optimizer, **{"T_max": 50, **scheduler_params}
            )


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_train_transform():
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(224),  # resize to 224 x 224 (required by ViT)
            transforms.ToTensor(),
            # Imagenet normalization
            # DINO model has learned features from ImageNet, so during fine-tuning on CIFAR-100,
            # the model will expect inputs to be normalized in the same way as during pretraining.
            # TODO check this is correct
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform


def get_test_tranform():
    transform = transforms.Compose(
        [
            transforms.Resize(224),  # resize to 224 x 224 (required by ViT)
            transforms.ToTensor(),
            # Imagenet normalization
            # DINO model has learned features from ImageNet, so during fine-tuning on CIFAR-100,
            # the model will expect inputs to be normalized in the same way as during pretraining.
            # TODO check this is correct
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform


def get_cifar_100_datasets() -> Tuple[Dataset, Dataset]:
    train_transform = get_train_transform()
    test_transform = get_test_tranform()
    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=test_transform
    )
    return trainset, testset


def get_cifar_100_train_valset_datasets(
    dataset: Dataset, seed: int = 42
) -> Tuple[Dataset, Dataset]:
    data_size = len(dataset)  # type: ignore
    train_size = int(0.8 * data_size)  # 80pc train, 20pc validation
    val_size = data_size - train_size
    trainset, valset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
    )
    return trainset, valset


def get_dataloader(
    dataset: Dataset,
    indices: Optional[List[int]] = None,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
) -> DataLoader:
    """
    Return a DataLoader for a given dataset, optionally using a subset of the data.

    Args:
        dataset (Dataset): The dataset to be loaded.
        indices (List[int], optional): A list of indices to create a subset of the dataset. If None, the entire dataset is used. Defaults to None.
        batch_size (int, optional): The number of samples per batch. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the data at the beginning of each epoch. Defaults to True.
            NB Should be set to False for test dataset and to True for train/val datasets.

    Returns:
        DataLoader: A DataLoader for the given dataset, potentially using a subset of the data.
    """
    batch_size = batch_size or 32  # if batch_size is None, use 32
    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices=indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


def get_cifar_dataloaders(batch_size=None):
    trainset, _ = get_cifar_100_datasets()
    trainset, valset = get_cifar_100_train_valset_datasets(trainset)
    train_dataloader, val_dataloader = get_dataloader(
        trainset, batch_size=batch_size
    ), get_dataloader(valset, batch_size=batch_size)
    return train_dataloader, val_dataloader

def _train(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    loss_func: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
):
    device = get_device()
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

        running_loss += loss.item()
        _, predicted = preds.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    if scheduler is not None:
        scheduler.step()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100.0 * correct / total

    return train_loss, train_accuracy


def compute_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[str] = None,
    loss_function: Optional[nn.Module] = None,
):
    """
    Compute predictions for a given dataloader using the trained model.

    Args:
        model: The trained model.
        dataloader: The DataLoader containing the test or train dataset.
        device: The device to use ('cpu' or 'cuda').
        loss_function: The loss function to minimize in core.

    Returns:
        predictions: Tensor of predictions.
        labels: Tensor of true labels.
        loss: Computed loss for the given data.
        accuracy: Computed accuracy for the given data.
    """
    model.eval()  # Set the model to evaluation mode
    predictions = []
    labels = []
    device = device or get_device()

    loss = 0.0

    # Disable gradient computation during inference
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(
                device
            )  # Move to the appropriate device

            # Forward pass
            preds = model(inputs)  # Get raw model predictions

            if loss_function is not None:
                loss += loss_function(preds, targets).item()

            # Get predicted class (class with the highest score)
            _, predicted = torch.max(preds, 1)

            predictions.append(predicted)
            labels.append(targets)

    # Concatenate all predictions and labels
    predictions = torch.cat(predictions)
    labels = torch.cat(labels)

    correct = (predictions == labels).sum().item()
    accuracy = 100.0 * correct / len(labels)
    loss = loss / len(labels)

    return predictions, labels, loss, accuracy


def train_model(
    *,
    training_params: TrainingParams,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    project_name: Optional[str] = None,
    wandb_log: bool = True,
    wandb_save: bool = True,
) -> Dict[str, Union[torch.nn.Module, float]]:
    """
    Train a pytorch-based model with the given training parameters and data loaders.

    Args:
        training_params (TrainingParams): Training parameters including model, optimizer, loss function, and scheduler.
        train_loader (DataLoader): PyTorch DataLoader for the training dataset.
        val_loader (Optional[DataLoader], optional): PyTorch DataLoader for the validation dataset, defaults to None.
        project_name (str, optional): Name of the WandB project, defaults to "mldl".
        wandb_log (bool): Use wandb to log training output, otherwise fallback to default logger. Defaults to True.
        wandb_save (bool): Use wandb to save trained model, otherwise do not save. Defaults to True.

    Returns:
        Dict[str, torch.nn.Module, float]: A dictionary containing the trained model and the best validation accuracy.
            - 'model' (torch.nn.Module): The trained model.
            - 'best_accuracy' (float): The highest validation accuracy achieved during training.
    """

    assert isinstance(training_params, TrainingParams)
    assert isinstance(train_loader, torch.utils.data.DataLoader)
    assert isinstance(val_loader, torch.utils.data.DataLoader)

    use_wandb = wandb_log or wandb_save
    if use_wandb:
        if project_name is None:
            raise ValueError(
                "project name cannot be None with either wandb_log or wandb_save set to True."
            )

    if use_wandb:
        wandb.init(
            project=project_name,
            name=training_params.training_name,
            config={
                "epochs": training_params.epochs,
                "batch_size": train_loader.batch_size,
                "learning_rate": training_params.learning_rate,
                "architecture": training_params.model.__class__.__name__,
                "optimizer_class": training_params.optimizer_class.__name__,
                "loss_function": training_params.loss_function.__class__.__name__,
                **(training_params.optimizer_params or {}),
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
        train_loss, train_accuracy = _train(
            model=model,
            train_loader=train_loader,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if wandb_log:
            # Log core metrics to wandb
            wandb.log(
                {
                    "Epoch": epoch,
                    "Train Loss": train_loss,
                    "Train Accuracy": train_accuracy,
                }
            )
        else:
            logging.info(
                f"Epoch: {epoch}: Train Loss: {train_loss}, Train Accuracy: {train_accuracy}"
            )

        if val_loader:
            _, _, val_loss, val_accuracy = compute_predictions(
                model=model, dataloader=val_loader, loss_function=loss_func
            )
            if wandb_log:
                wandb.log(
                    {
                        "Epoch": epoch,
                        "Validation Loss": val_loss,
                        "Validation Accuracy": val_accuracy,
                    }
                )
            else:
                logging.info(
                    f"Epoch: {epoch}: Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}"
                )

            # Save the model with the best validation accuracy
            if val_accuracy > best_acc:
                best_acc = val_accuracy
                if wandb_save:
                    model_name = f"{training_params.training_name}_best.pth"
                    torch.save(model.state_dict(), model_name)
                    wandb.save(model_name)

    if use_wandb:
        wandb.finish()

    res_dict = {"model": model, "best_accuracy": best_acc}
    return res_dict





def run_single(*, lr=1e-3, momentum=0.9, weight_decay=5e-4, batch_size=64):
    train_dataloader, val_dataloader = get_cifar_dataloaders(batch_size=batch_size)
    dino = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
    dino.head = nn.Linear(384, 100)
    params = TrainingParams(
        training_name=f"centralized_baseline_bs_{batch_size}_momentum_{momentum:.2f}_wdecay_{weight_decay:.2f}_lr_{lr:.2f}_cosineLR",
        model=dino,
        loss_function=nn.CrossEntropyLoss(),
        learning_rate=lr,
        optimizer_class=torch.optim.SGD,  # type: ignore
        scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,  # type: ignore
        epochs=10,
        optimizer_params={"momentum": momentum, "weight_decay": weight_decay},
        scheduler_params={"T_max": 20},
    )
    res_dict = train_model(
        training_params=params,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        project_name="fl_centralized_baseline",
    )
    return res_dict["best_accuracy"]

if __name__ == "__main__":
    # optimize()
    run_single()
