

class Configuration:

    def __init__(self, clients_number: int, 
                 servers_number: int,
                 aggregation_method: str,
                 local_epochs: int,
                 batch_size: int,
                 learning_rate: float,
                 momentum: float,
                 weight_decay: float,
                 dataset: str,
                 optimizer: str,
                 data_partition_method: str,
                 loss_function: str):

        self.clients_number = clients_number
        self.servers_number = servers_number 
        self.aggregation_method = aggregation_method
        self.local_epochs = local_epochs        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.dataset = dataset
        self.optimizer = optimizer
        self.data_partition_method = data_partition_method
        self.optimizer = self.optimizer 
        self.data_partition_method = self.data_partition_method
        self.loss_function = loss_function




