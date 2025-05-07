

class Configuration:
    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 momentum: float,
                 weight_decay: float,
                 dataset: str, # Changed from dataset_name
                 optimizer: str,
                 loss_function: str):

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dataset = dataset # Changed from dataset_name
        self.optimizer = optimizer
        self.loss_function = loss_function




