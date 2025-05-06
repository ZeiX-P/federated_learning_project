from config import Config

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split

from typing import List, Tuple, Optional, DataSet


class Dataset:

    def __init__(self):

        self.number_of_clients = Config.clients_number
        self.number_of_servers = Config.servers_number
        
        self.trasform_train = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        self.trasform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        


    def get_CIFAR_dataset(self, apply_transform: bool = True) -> tuple[Dataset, Dataset]:
        """
        Load the dataset CIFAR100
        """
        if apply_transform:
            train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=None)
            
            val_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=None)
        else:
            train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=self.trasform_train)
            
            val_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=self.trasform_test)
        return train_set, val_set
    
    def get_dataloader(self, dataset: Dataset, indices: Optional[List[int]] = None) -> DataLoader:

        """
        Create a dataloader for the dataset
        """
        if indices is not None:
            dataset = torch.utils.data.Subset(dataset, indices)
        dataloader = DataLoader(dataset, batch_size=Config.batch_size,
                                shuffle=True, num_workers=2)
        return dataloader
    
    