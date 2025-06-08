
from collections import defaultdict
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
                                shuffle=True)
        
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
    
    def non_iid_sharding(self,
    dataset: Dataset,
    num_clients: int,
    seed: Optional[int] = 42,
) -> Dict[int, List[int]]:
        """
        Split the dataset into non-i.i.d. shards.

        Each client receives samples from exactly `num_classes` classes.

        Args:
            dataset (Dataset): Dataset to be split.
            num_clients (int): Number of clients.
            num_classes (int): Number of classes per client.
            seed (Optional[int]): Random seed.

        Returns:
            Dict[int, List[int]]: Mapping client ID to list of sample indices.
        """
        client_data = defaultdict(list)
        class_indices = defaultdict(list)
        num_classes = 10
        # Group sample indices by class
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)

        all_classes = list(class_indices.keys())
        total_classes = len(all_classes)

        if num_classes > total_classes:
            raise ValueError(
                f"Requested {num_classes} classes per client, "
                f"but dataset only has {total_classes} classes."
            )

        if num_clients * num_classes > total_classes * len(class_indices[0]):
            print("Warning: There may be overlapping class assignments among clients.")

        rng = np.random.default_rng(seed)

        # Shuffle class list for randomness
        rng.shuffle(all_classes)

        # Assign classes to clients
        class_pool = all_classes.copy()
        for client_id in range(num_clients):
            if len(class_pool) < num_classes:
                class_pool = all_classes.copy()
                rng.shuffle(class_pool)
            selected_classes = class_pool[:num_classes]
            class_pool = class_pool[num_classes:]

            # Assign samples from selected classes
            for cls in selected_classes:
                samples = class_indices[cls]
                rng.shuffle(samples)
                client_data[client_id].extend(samples)

        return dict(client_data)



    def non_iid_split_by_class_dataset(self, dataset: Dataset, num_clients: int, num_classes_per_client: int = 2):
        """
        Splits a dataset into non-IID subsets based on class labels for Federated Learning.
        Handles both base Dataset objects and Subset objects.

        Args:
            dataset (Dataset): The dataset object (e.g., a PyTorch Dataset or Subset) from which
                               to get labels and indices.
            num_clients (int): The number of clients to split the data among.
            num_classes_per_client (int): The maximum number of distinct classes
                                          each client will primarily receive data from.

        Returns:
            dict: A dictionary where keys are client IDs (e.g., 'client_0', 'client_1')
                  and values are lists of data indices assigned to that client.
                  These indices are relative to the *original dataset* if `dataset` is a Subset,
                  or relative to `dataset` itself if it's a base Dataset.
        """
        # --- Crucial change here to correctly get labels ---
        # If it's a Subset, the actual labels are in the base dataset
        if isinstance(dataset, Subset):
            base_dataset = dataset.dataset
            subset_indices = dataset.indices
            # Get labels corresponding to the subset's indices from the base dataset
            dataset_labels = np.array([base_dataset.targets[i] for i in subset_indices])
            # The indices we'll distribute are the `subset_indices` themselves
            # because they map back to the original dataset.
            all_dataset_indices = list(subset_indices)
        else: # Assume it's a base Dataset (e.g., torchvision.datasets.CIFAR10)
            dataset_labels = np.array(dataset.targets)
            all_dataset_indices = list(range(len(dataset))) # These are the indices we're working with

        total_samples = len(all_dataset_indices)
        unique_classes = np.unique(dataset_labels)
        num_total_classes = len(unique_classes)

        if num_classes_per_client > num_total_classes:
            print(f"Warning: num_classes_per_client ({num_classes_per_client}) is greater than "
                  f"total unique classes ({num_total_classes}). Setting to {num_total_classes}.")
            num_classes_per_client = num_total_classes

        # Step 1: Group indices (from `all_dataset_indices`) by class
        # We need to map the labels back to the original indices from `all_dataset_indices`
        class_indices_map = defaultdict(list)
        for idx_in_labels_array, label in enumerate(dataset_labels):
            # The index in dataset_labels corresponds to an index in all_dataset_indices
            original_data_index = all_dataset_indices[idx_in_labels_array]
            class_indices_map[label].append(original_data_index)


        # Step 2: Distribute classes to clients
        client_data_indices = defaultdict(list)
        
        # Shuffle unique classes to ensure a more random initial assignment
        shuffled_classes = np.random.permutation(unique_classes)

        # Distribute classes to clients in a round-robin fashion, ensuring non-IID
        class_idx_pointer = 0
        for client_id in range(num_clients):
            for _ in range(num_classes_per_client):
                if class_idx_pointer >= num_total_classes:
                    # If we've run out of classes to assign, cycle back
                    class_idx_pointer = 0
                
                current_class = shuffled_classes[class_idx_pointer]
                
                # Add all indices for this class to the current client
                client_data_indices[client_id].extend(class_indices_map[current_class])
                
                # Clear the assigned indices from the map to avoid re-assigning them primarily
                class_indices_map[current_class] = [] 
                
                class_idx_pointer += 1

        # Step 3: Handle any remaining data (classes that might not have been fully assigned)
        remaining_indices = []
        for cls in unique_classes:
            remaining_indices.extend(class_indices_map[cls]) # Collect any left-over indices

        # Distribute remaining indices randomly among clients
        if remaining_indices:
            np.random.shuffle(remaining_indices)
            for i, idx in enumerate(remaining_indices):
                client_id = i % num_clients
                client_data_indices[client_id].append(idx)

        # Format the output dictionary with 'client_X' keys
        final_client_splits = {}
        for i in range(num_clients):
            np.random.shuffle(client_data_indices[i])
            final_client_splits[f'client_{i}'] = client_data_indices[i]

        return final_client_splits



    