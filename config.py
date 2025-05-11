import torch
import torch.nn as nn

class Configuration:
    def __init__(self,
                 model: nn.Module,
                 training_name: str,
                 batch_size: int,
                 learning_rate: float,
                 momentum: float,
                 weight_decay: float,
                 dataset: str, # Changed from dataset_name
                 optimizer: torch.optim.Optimizer,
                 loss_function: nn.Module,
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 epochs: int,
                 optimizer_params,
                 scheduler_params,
                 ):
        self.model = model
        self.training_name = training_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dataset = dataset # Changed from dataset_name
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.epochs = epochs
        self.optimizer_params = optimizer_params 
        self.scheduler_params = scheduler_params


    @property
    def optimizer(self):
        optimizer_params = self.optimizer_params or {}
        return self.optimizer_class(
            self.model.parameters(), lr=self.learning_rate, **optimizer_params
        )

    @property
    def scheduler(self):
        scheduler_params = self.scheduler_params or {}
        if self.scheduler:
            return self.scheduler(
                self.optimizer, **{"T_max": 50, **scheduler_params}
            )




