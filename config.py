import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional, Dict

class Configuration:
    def __init__(
        self,
        model: nn.Module,
        training_name: str,
        batch_size: int,
        learning_rate: float,
        momentum: float,
        weight_decay: float,
        dataset: str,
        loss_function: nn.Module,
        epochs: int,
        project_name,
        optimizer_class: type = optim.Adam,
        scheduler_class: Optional[type] = None,
        optimizer_params: Optional[Dict] = None,
        scheduler_params: Optional[Dict] = None,
        
    ):
        self.model = model
        self.training_name = training_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dataset = dataset
        self.loss_function = loss_function
        self.epochs = epochs
        self.optimizer_class = optimizer_class
        self.scheduler_class = scheduler_class
        self.optimizer_params = optimizer_params or {}
        self.scheduler_params = scheduler_params or {}
        self.project_name = project_name

        # Now correctly store instances, not class names
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()

    def _init_optimizer(self):
        """Initializes the optimizer."""
        base_params = {
            'params': self.model.parameters(),
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay,
        }

        if self.optimizer_class == optim.SGD:
            if 'momentum' not in self.optimizer_params:
                base_params['momentum'] = self.momentum

            # Exclude Adam-specific parameters
            filtered_params = {
                k: v for k, v in self.optimizer_params.items()
                if k not in ['amsgrad', 'betas', 'eps']
            }
            base_params.update(filtered_params)
        else:
            base_params.update(self.optimizer_params)

        return self.optimizer_class(**base_params)

    def _init_scheduler(self) -> Optional[LRScheduler]:
        """Initializes the learning rate scheduler."""
        if self.scheduler_class is None:
            return None

        scheduler_params = self.scheduler_params.copy()
        if 'T_max' not in scheduler_params and self.scheduler_class.__name__ == 'CosineAnnealingLR':
            scheduler_params['T_max'] = self.epochs

        return self.scheduler_class(self.optimizer, **scheduler_params)

'''
import attr
import torch
from torch import nn
from typing import Optional, Dict, Any
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from optim.ssgd import SparseSGDM


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

    def __attrs_post_init__(self):
        if self.optimizer_class is SparseSGDM:
            if not isinstance(self.optimizer_params, dict):
                raise ValueError(
                    "optimizer_params must be a dictionary when using SparseSGDM."
                )

            if "named_params" not in self.optimizer_params:
                raise ValueError(
                    "SparseSGDM requires 'named_params' in optimizer_params."
                )

            if "grad_mask" not in self.optimizer_params:
                raise ValueError("SparseSGDM requires 'grad_mask' in optimizer_params.")

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
'''