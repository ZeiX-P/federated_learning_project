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
                 epochs: int):
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




