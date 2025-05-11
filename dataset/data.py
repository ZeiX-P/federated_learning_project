
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset

from typing import List, Tuple, Optional, Dict

import numpy as np
import random

class Dataset:

    def __init__(self):

        
        self.trasform_train = transforms.Compose(
            [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.Resize((224,224)),  # resize to 224 x 224 (required by ViT)
            transforms.ToTensor(),
            # Imagenet normalization
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
            )
        
        self.trasform_test = transforms.Compose(
            [
            transforms.Resize(224),  # resize to 224 x 224 (required by ViT)
            transforms.ToTensor(),
            # Imagenet normalization
            # DINO model has learned features from ImageNet, so during fine-tuning on CIFAR-100,
            # the model will expect inputs to be normalized in the same way as during pretraining.
            # TODO check this is correct
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    

    def create_train_val_set(self, dataset: Dataset, seed: int = 42):

        data_size = len(dataset)  
        train_size = int(0.8 * data_size)  # 80pc train, 20pc validation
        val_size = data_size - train_size
        trainset, valset = random_split(dataset, [train_size, val_size])

        return trainset, valset   

    def get_dataset(self, dataset_name:str, apply_transform: bool = True) -> tuple[Dataset, Dataset]:
        """
        Load dataset from torchvision and apply transformations if needed.
        """

        dataset_class = getattr(torchvision.datasets, dataset_name)

        if apply_transform:
            train_set = dataset_class(root='./data', train=True,
                                                download=True, transform=self.trasform_train)
            
            val_set = dataset_class(root='./data', train=False,
                                                download=True, transform=self.trasform_test)
        else:
            train_set = dataset_class(root='./data', train=True,
                                                download=True, transform=None)
            
            val_set = dataset_class(root='./data', train=False,
                                                download=True, transform=None)
        return train_set, val_set
    
    def get_dataloader(self, dataset: Dataset, indices: Optional[List[int]] = None) -> DataLoader:

        """
        Create a dataloader for the dataset
        """
        if indices is not None:
            dataset,_ = torch.utils.data.Subset(dataset, indices)
        dataloader = DataLoader(dataset, batch_size=64,
                                shuffle=True, num_workers=2)
        return dataloader
    
    def get_dataloaders(self,dataset_name: str):

        dataset,_ = self.get_dataset(dataset_name, apply_transform=True)
        train_set, val_set = self.create_train_val_set(dataset)
        
        train_loader = DataLoader(train_set, batch_size=64,
                                shuffle=True)
        
        val_loader = DataLoader(val_set, batch_size=64,
                                shuffle=False)
        
        return train_loader, val_loader
    
    def create_federated_datasets(dataset: Dataset, indices_dict: Dict[str, List[int]]) -> Dict[str, Subset]:
        
        federated_datasets: Dict[str, Subset] = {}
        for client_id, indices in indices_dict.items():
            federated_datasets[client_id] = torch.utils.data.Subset(dataset, indices)
        return federated_datasets

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

    