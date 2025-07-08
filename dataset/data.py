
from collections import defaultdict
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import logging
from typing import List, Tuple, Optional, Dict

import numpy as np
import random
import wandb
from collections import defaultdict
from torch.utils.data import Dataset, Subset # Import Dataset and Subset
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from typing import Optional, List, Dict
import random

class Dataset: # Keeping the class name as 'Dataset' as per your provided code
    def __init__(self):
        self.trasform_train = transforms.Compose(
            [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        self.trasform_test = transforms.Compose(
            [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
    
    def create_train_val_set(self, dataset: Dataset, seed: int = 42):
        data_size = len(dataset)
        train_size = int(0.8 * data_size)
        val_size = data_size - train_size
        
        generator = torch.Generator().manual_seed(seed) # Ensure reproducibility
        trainset, valset = random_split(dataset, [train_size, val_size], generator=generator)

        return trainset, valset 

    def get_dataset(self, dataset_name:str, apply_transform: bool = True) -> tuple[Dataset, Dataset]:
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
        if indices is not None:
            dataset = torch.utils.data.Subset(dataset, indices) # Fixed: no _,
        dataloader = DataLoader(dataset, batch_size=64,
                                 shuffle=True, num_workers=2)
        return dataloader
    
    def get_dataloaders(self,dataset_name: str):
        dataset_full, _ = self.get_dataset(dataset_name, apply_transform=True)
        train_set, val_set = self.create_train_val_set(dataset_full)
        
        train_loader = DataLoader(train_set, batch_size=64,
                                 shuffle=True)
        
        val_loader = DataLoader(val_set, batch_size=64,
                                 shuffle=True)
        
        return train_loader, val_loader
    
    def create_federated_datasets(self, dataset: Dataset, indices_dict: Dict[int, List[int]]) -> Dict[int, Subset]:
        federated_datasets: Dict[int, Subset] = {}
        for client_id, indices in indices_dict.items():
            federated_datasets[client_id] = torch.utils.data.Subset(dataset, indices)
        return federated_datasets

    def idd_split(self, dataset: Dataset, num_clients: int, seed: int = 42) -> Dict[int, List[int]]:
        """
        Create a list of indices for each client (IID split).
        Returns: Dict with integer keys.
        """
        # Set seeds for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        indices_clients = {i: [] for i in range(num_clients)}
        indices = list(range(len(dataset)))
        
        np.random.shuffle(indices)
        for i in range(len(dataset)):
            client_id = i % num_clients
            indices_clients[client_id].append(indices[i])
        # Distribute indices round-robin style for perfect balance
        '''
        for idx, data_idx in enumerate(indices):
            client_id = idx % num_clients
            indices_clients[client_id].append(data_idx)
        '''

        
        
        return indices_clients
        
    
    
    def dirichlet_non_iid_split(self, dataset: Dataset, num_clients: int, alpha: float = 0.5, seed: int = 42) -> Dict[int, List[int]]:
    
        np.random.seed(seed)
        
        # Handle both regular Dataset and Subset objects
        if isinstance(dataset, torch.utils.data.Subset):
            # This is the case after create_train_val_set
            base_dataset = dataset.dataset
            subset_indices = dataset.indices
            # Get labels for the subset
            labels = np.array([base_dataset.targets[i] for i in subset_indices])
            total_samples = len(subset_indices)
        else:
            # Regular dataset
            labels = np.array(dataset.targets)
            total_samples = len(dataset)
        
        print(f"Splitting {total_samples} samples across {num_clients} clients with alpha={alpha}")
        
        # Group sample indices by class
        class_to_indices = defaultdict(list)
        for sample_idx, label in enumerate(labels):
            # sample_idx is 0-based index within our current dataset/subset
            class_to_indices[label].append(sample_idx)
        
        unique_classes = list(class_to_indices.keys())
        print(f"Found {len(unique_classes)} classes: {unique_classes}")
        
        # Initialize client data containers
        client_indices = {client_id: [] for client_id in range(num_clients)}
        
        # Distribute each class across clients using Dirichlet distribution
        for class_label in unique_classes:
            class_indices = class_to_indices[class_label]
            np.random.shuffle(class_indices)
            
            # Generate proportions for this class across all clients
            proportions = np.random.dirichlet([alpha] * num_clients)
            
            # Ensure proportions sum to 1
            proportions = proportions / proportions.sum()
            
            # Calculate split points
            split_lengths = (proportions * len(class_indices)).astype(int)
            
            # Adjust for rounding errors - give remaining samples to random clients
            remaining = len(class_indices) - split_lengths.sum()
            if remaining > 0:
                random_clients = np.random.choice(num_clients, remaining, replace=False)
                for client_id in random_clients:
                    split_lengths[client_id] += 1
            
            # Distribute indices to clients
            start_idx = 0
            for client_id in range(num_clients):
                end_idx = start_idx + split_lengths[client_id]
                if end_idx > start_idx:  # Only add if there are samples to add
                    client_indices[client_id].extend(class_indices[start_idx:end_idx])
                start_idx = end_idx
        
        # Shuffle each client's data and validate
        for client_id in range(num_clients):
            if client_indices[client_id]:  # Only shuffle if not empty
                np.random.shuffle(client_indices[client_id])
                
                # Validation - ensure all indices are valid
                max_idx = max(client_indices[client_id])
                min_idx = min(client_indices[client_id])
                
                if max_idx >= total_samples or min_idx < 0:
                    raise ValueError(f"Client {client_id} has invalid indices. "
                                f"Range: [{min_idx}, {max_idx}], Dataset size: {total_samples}")
        
        # Print distribution summary
        for client_id in range(num_clients):
            print(f"Client {client_id}: {len(client_indices[client_id])} samples")
        
        return client_indices

    def pathological_non_iid_split(self, dataset: Dataset, num_clients: int, classes_per_client: int = 10, seed: int = 42) -> Dict[int, List[int]]:
        """
        Pathological Non-IID split where each client gets samples from exactly 'classes_per_client' classes.
        
        Args:
            dataset: The dataset to split
            num_clients: Number of clients
            classes_per_client: Number of classes each client should have (Nc parameter)
            seed: Random seed for reproducibility
        
        Returns:
            Dictionary mapping client_id to list of sample indices
        """
        np.random.seed(seed)
        
        # Handle both regular Dataset and Subset objects
        if isinstance(dataset, torch.utils.data.Subset):
            base_dataset = dataset.dataset
            subset_indices = dataset.indices
            labels = np.array([base_dataset.targets[i] for i in subset_indices])
            total_samples = len(subset_indices)
            # Use indices relative to the subset (0-based)
            working_indices = list(range(total_samples))
        else:
            labels = np.array(dataset.targets)
            total_samples = len(dataset)
            working_indices = list(range(total_samples))
        
        logging.info(f"Pathological Non-IID: Splitting {total_samples} samples across {num_clients} clients")
        logging.info(f"Classes per client: {classes_per_client}")
        
        # Step 1: Group samples by class
        class_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_to_indices[label].append(working_indices[idx])
        
        unique_classes = sorted(class_to_indices.keys())
        num_classes = len(unique_classes)
        
        logging.info(f"Found {num_classes} classes: {unique_classes}")
        for class_label in unique_classes:
            logging.info(f"  Class {class_label}: {len(class_to_indices[class_label])} samples")
        
        # Validate that we can satisfy the requirement
        if classes_per_client > num_classes:
            raise ValueError(f"Cannot assign {classes_per_client} classes per client when only {num_classes} classes exist")
        
        # Step 2: Assign classes to clients
        # We need to ensure each client gets exactly 'classes_per_client' classes
        # and classes are distributed as evenly as possible across clients
        
        total_class_assignments = num_clients * classes_per_client
        classes_needed = total_class_assignments
        
        # Create a list of class assignments - each class appears multiple times
        class_assignments = []
        classes_per_round = classes_needed // num_classes
        remaining_assignments = classes_needed % num_classes
        
        # Assign each class 'classes_per_round' times
        for class_label in unique_classes:
            class_assignments.extend([class_label] * classes_per_round)
        
        # Assign remaining classes randomly
        if remaining_assignments > 0:
            remaining_classes = np.random.choice(unique_classes, remaining_assignments, replace=False)
            class_assignments.extend(remaining_classes)
        
        # Shuffle the class assignments
        np.random.shuffle(class_assignments)
        
        # Step 3: Assign classes to clients
        client_classes = {client_id: [] for client_id in range(num_clients)}
        
        assignment_idx = 0
        for client_id in range(num_clients):
            for _ in range(classes_per_client):
                client_classes[client_id].append(class_assignments[assignment_idx])
                assignment_idx += 1
        
        logging.info("Class assignments per client:")
        for client_id in range(num_clients):
            logging.info(f"  Client {client_id}: classes {sorted(client_classes[client_id])}")
        
        # Step 4: Distribute samples within each class to clients that have that class
        client_indices = {client_id: [] for client_id in range(num_clients)}
        
        for class_label in unique_classes:
            class_indices = class_to_indices[class_label].copy()
            np.random.shuffle(class_indices)
            
            # Find which clients have this class
            clients_with_class = [client_id for client_id in range(num_clients) 
                                if class_label in client_classes[client_id]]
            
            if not clients_with_class:
                logging.warning(f"No clients assigned to class {class_label}")
                continue
            
            # Count how many times each client appears for this class
            client_class_counts = {client_id: client_classes[client_id].count(class_label) 
                                for client_id in clients_with_class}
            
            total_shares = sum(client_class_counts.values())
            
            # Distribute samples proportionally
            start_idx = 0
            for client_id in clients_with_class:
                share_ratio = client_class_counts[client_id] / total_shares
                num_samples = int(len(class_indices) * share_ratio)
                
                # Handle the last client to get remaining samples
                if client_id == clients_with_class[-1]:
                    num_samples = len(class_indices) - start_idx
                
                if num_samples > 0:
                    client_indices[client_id].extend(class_indices[start_idx:start_idx + num_samples])
                    start_idx += num_samples
        
        # Step 5: Shuffle each client's data
        for client_id in range(num_clients):
            if client_indices[client_id]:
                np.random.shuffle(client_indices[client_id])
        
        # Step 6: Analyze the results
        print(f"\nFinal distribution (each client should have exactly {classes_per_client} classes):")
        total_assigned = 0
        
        for client_id in range(num_clients):
            if client_indices[client_id]:
                # Get labels for this client's samples
                if isinstance(dataset, torch.utils.data.Subset):
                    client_labels = [labels[idx] for idx in client_indices[client_id]]
                else:
                    client_labels = [dataset.targets[idx] for idx in client_indices[client_id]]
                
                unique_client_labels, client_counts = np.unique(client_labels, return_counts=True)
                label_dist = {int(label): int(count) for label, count in zip(unique_client_labels, client_counts)}
                
                print(f"  Client {client_id}: {len(unique_client_labels)} classes, {len(client_indices[client_id])} samples")
                print(f"    Classes: {label_dist}")
                
                total_assigned += len(client_indices[client_id])
                
                # Verify the client has exactly the right number of classes
                if len(unique_client_labels) != classes_per_client:
                    logging.warning(f"Client {client_id} has {len(unique_client_labels)} classes, expected {classes_per_client}")
        
        print(f"\nValidation: {total_assigned} samples assigned out of {total_samples} total")
        
        # Additional validation
        for client_id in range(num_clients):
            if client_indices[client_id]:
                max_idx = max(client_indices[client_id])
                min_idx = min(client_indices[client_id])
                if max_idx >= total_samples or min_idx < 0:
                    raise ValueError(f"Client {client_id} has invalid indices. "
                                f"Range: [{min_idx}, {max_idx}], Dataset size: {total_samples}")
        
        return client_indices

