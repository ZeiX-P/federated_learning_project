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
