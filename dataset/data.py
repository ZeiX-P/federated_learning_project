from config import Config

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split

from typing import List, Tuple, Optional, Dict

import numpy as np
import random

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
    
    def idd_split(self, dataset: Dataset, num_clients: int) -> Dict[int, List[int]]:

        """
        Create a list of indices for each client
        """

        indices_clients = {i:[] for i in range(num_clients)}
        indices = list(range(len(dataset)))
        data_per_client = len(dataset) // num_clients

        np.random.shuffle(indices)

        for i in range(num_clients):

            start_idx = i * data_per_client
            end_idx = (i + 1) * data_per_client
            client_indices = indices[start_idx:end_idx]

            indices_clients[i] = client_indices

        if end_idx < len(dataset):

            remaining_indices = indices[end_idx:]

            for i in range(len(remaining_indices)):

                indices_clients[random.randrange(num_clients)].append(remaining_indices[i])

        return indices_clients
  
    def dirichlet_non_iid_split(self, dataset: Dataset, num_clients, alpha=0.5, seed=42):
        np.random.seed(seed)
 
        client_data = {}

        # Group indices of each class
        class_indices = {}
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)

        labels_classes = np.unique(list(class_indices.keys()))

        for label in labels_classes:
            indices = class_indices[label]
            np.random.shuffle(indices)

            # Sample proportions for each client from Dirichlet distribution
            proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
            proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
            splits = np.split(indices, proportions)

            for client_id, split in enumerate(splits):
                client_data[client_id].extend(split)

        return client_data
