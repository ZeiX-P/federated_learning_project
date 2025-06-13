
from collections import defaultdict
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import logging
from typing import List, Tuple, Optional, Dict

import numpy as np
import random
'''
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



    import numpy as np
'''
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

    def idd_split(self, dataset: Dataset, num_clients: int) -> Dict[int, List[int]]:
        """
        Create a list of indices for each client (IID split).
        Returns: Dict with integer keys.
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

    def pathological_non_iid_split1(self, dataset: Dataset, num_clients: int, shards_per_client: int = 2, seed: int = 42) -> Dict[int, List[int]]:
    
        np.random.seed(seed)
        
        # Handle both regular Dataset and Subset objects
        if isinstance(dataset, torch.utils.data.Subset):
            base_dataset = dataset.dataset
            subset_indices = dataset.indices
            # Get labels for the subset
            labels = np.array([base_dataset.targets[i] for i in subset_indices])
            total_samples = len(subset_indices)
            # Use sequential indices for the subset (0, 1, 2, ..., len(subset)-1)
            working_indices = list(range(total_samples))
        else:
            # Regular dataset
            labels = np.array(dataset.targets)
            total_samples = len(dataset)
            working_indices = list(range(total_samples))
        
        logging.info(f"Pathological Non-IID: Splitting {total_samples} samples across {num_clients} clients")
        logging.info(f"Shards per client: {shards_per_client}")
        
        # Step 1: Sort data by labels
        # Create (label, working_index) pairs and sort by label
        label_index_pairs = [(labels[i], working_indices[i]) for i in range(total_samples)]
        label_index_pairs.sort(key=lambda x: x[0])  # Sort by label
        
        # Extract sorted indices
        sorted_indices = [pair[1] for pair in label_index_pairs]
        sorted_labels = [pair[0] for pair in label_index_pairs]
        
        logging.info(f"Data sorted by labels. Label distribution after sorting:")
        unique_labels, counts = np.unique(sorted_labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            logging.info(f"  Label {label}: {count} samples")
        
        # Step 2: Calculate shard configuration
        total_shards = num_clients * shards_per_client
        shard_size = total_samples // total_shards
        
        if total_samples % total_shards != 0:
            logging.warning(f"{total_samples} samples cannot be evenly divided into {total_shards} shards.")
            logging.warning(f"Using shard size of {shard_size}, leaving {total_samples % total_shards} samples unused.")
        
        logging.info(f"Creating {total_shards} shards of size {shard_size} each")
        
        # Step 3: Create shards from sorted data
        shards = []
        for shard_id in range(total_shards):
            start_idx = shard_id * shard_size
            end_idx = start_idx + shard_size
            if end_idx <= len(sorted_indices):
                shard_indices = sorted_indices[start_idx:end_idx]
                shards.append(shard_indices)
            else:
                # Handle case where we don't have enough samples for the last shard
                break
        
        actual_shards = len(shards)
        print(f"Created {actual_shards} complete shards")
        
        # Step 4: Randomly assign shards to clients
        # Create list of shard IDs and shuffle them
        shard_ids = list(range(actual_shards))
        np.random.shuffle(shard_ids)
        
        # Initialize client data containers
        client_indices = {client_id: [] for client_id in range(num_clients)}
        
        # Assign shards to clients
        shard_idx = 0
        for client_id in range(num_clients):
            client_shards = []
            for _ in range(shards_per_client):
                if shard_idx < len(shard_ids):
                    shard_id = shard_ids[shard_idx]
                    client_indices[client_id].extend(shards[shard_id])
                    client_shards.append(shard_id)
                    shard_idx += 1
            
            # Shuffle the indices within each client to avoid ordering bias
            if client_indices[client_id]:
                np.random.shuffle(client_indices[client_id])
            
            print(f"Client {client_id}: {len(client_indices[client_id])} samples from shards {client_shards}")
        
        # Step 5: Analyze the class distribution for each client
        print("\nClass distribution per client:")
        for client_id in range(num_clients):
            if client_indices[client_id]:
                # Get labels for this client's samples
                if isinstance(dataset, torch.utils.data.Subset):
                    # The indices in client_indices[client_id] are subset indices (0-based)
                    # We need to map them to the actual labels
                    client_labels = [labels[idx] for idx in client_indices[client_id]]
                else:
                    client_labels = [dataset.targets[idx] for idx in client_indices[client_id]]
                
                unique_client_labels, client_counts = np.unique(client_labels, return_counts=True)
                label_dist = {int(label): int(count) for label, count in zip(unique_client_labels, client_counts)}
                print(f"  Client {client_id}: {label_dist} (total: {len(client_indices[client_id])})")
        
        # Validation
        total_assigned = sum(len(indices) for indices in client_indices.values())
        print(f"\nValidation: {total_assigned} samples assigned out of {total_samples} total")
        
        # Additional validation for subset case
        if isinstance(dataset, torch.utils.data.Subset):
            for client_id in range(num_clients):
                if client_indices[client_id]:
                    max_idx = max(client_indices[client_id])
                    min_idx = min(client_indices[client_id])
                    if max_idx >= total_samples or min_idx < 0:
                        raise ValueError(f"Client {client_id} has invalid indices. "
                                    f"Range: [{min_idx}, {max_idx}], Dataset size: {total_samples}")
        
        return client_indices
    
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

# Optional: Add this method to your Dataset class for better debugging
    def validate_client_data_distribution(self, dict_client_data):
        """
        Validate and print information about client data distribution
        """
        print("\n=== Client Data Distribution Analysis ===")
        
        total_samples = 0
        class_distribution = defaultdict(lambda: defaultdict(int))
        
        for client_id, client_dataset in dict_client_data.items():
            client_size = len(client_dataset)
            total_samples += client_size
            
            # Analyze class distribution
            if hasattr(client_dataset, 'indices') and hasattr(client_dataset.dataset, 'targets'):
                try:
                    # Get labels for this client's data
                    indices = client_dataset.indices
                    if hasattr(client_dataset.dataset, 'dataset'):
                        # Handle nested Subset (train/val split from original dataset)
                        base_targets = client_dataset.dataset.dataset.targets
                        subset_indices = client_dataset.dataset.indices
                        targets = [base_targets[subset_indices[i]] for i in indices]
                    else:
                        targets = [client_dataset.dataset.targets[i] for i in indices]
                    
                    # Count classes
                    for target in targets:
                        class_distribution[client_id][target] += 1
                        
                except Exception as e:
                    print(f"Could not analyze client {client_id}: {e}")
        
        print(f"Total samples distributed: {total_samples}")
        
        # Print per-client class distribution
        for client_id in sorted(class_distribution.keys()):
            class_counts = class_distribution[client_id]
            print(f"Client {client_id}: {dict(sorted(class_counts.items()))}")
        
        return class_distribution
    
    '''
    def dirichlet_non_iid_split(self, dataset: Dataset, num_clients: int, alpha: float = 0.5, seed: int = 42) -> Dict[int, List[int]]:
        np.random.seed(seed)
        client_data = defaultdict(list)

        if isinstance(dataset, Subset):
            base_dataset = dataset.dataset
            subset_indices = dataset.indices
            dataset_labels = np.array([base_dataset.targets[i] for i in subset_indices])
            # Map subset indices to original dataset indices
            index_mapping = {i: subset_indices[i] for i in range(len(subset_indices))}
        else:
            dataset_labels = np.array(dataset.targets)
            index_mapping = {i: i for i in range(len(dataset_labels))}

        class_indices_in_dataset_scope = defaultdict(list)
        for idx_in_dataset, label in enumerate(dataset_labels):
            class_indices_in_dataset_scope[label].append(idx_in_dataset)

        labels_classes = np.unique(dataset_labels)

        for label in labels_classes:
            indices_for_label = class_indices_in_dataset_scope[label]
            np.random.shuffle(indices_for_label)

            proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
            proportions = proportions / proportions.sum()

            proportions_cumsum = (np.cumsum(proportions) * len(indices_for_label)).astype(int)
            proportions_cumsum[-1] = len(indices_for_label)

            current_idx = 0
            for client_id in range(num_clients):
                start_idx = current_idx
                end_idx = proportions_cumsum[client_id]
                split = indices_for_label[start_idx:end_idx]
                # Convert to appropriate indices based on dataset type
                mapped_indices = [index_mapping[idx] for idx in split]
                client_data[client_id].extend(mapped_indices)
                current_idx = end_idx

        final_client_splits = {}
        for i in range(num_clients):
            np.random.shuffle(client_data[i])
            final_client_splits[i] = client_data[i]

        return final_client_splits
    '''
    '''
    def non_iid_sharding(self,
        dataset: Dataset,
        num_clients: int,
        num_classes: int = 2, # Number of classes assigned to each client initially
        samples_per_class_shard: int = 10, # Number of samples to take from each class per shard
        seed: Optional[int] = 42,
    ) -> Dict[int, List[int]]:
        """
        Split the dataset into non-i.i.d. shards.
        Each client receives `samples_per_class_shard` from `num_classes` classes.
        Returns: Dict with integer keys.
        """
        client_data = defaultdict(list)
        
        if isinstance(dataset, Subset):
            base_dataset = dataset.dataset
            subset_indices = dataset.indices
            dataset_labels = np.array([base_dataset.targets[i] for i in subset_indices])
            all_dataset_original_indices = list(subset_indices)
        else:
            dataset_labels = np.array(dataset.targets)
            all_dataset_original_indices = list(range(len(dataset)))

        all_data_points = [(all_dataset_original_indices[i], label) for i, label in enumerate(dataset_labels)]
        
        class_indices_pool = defaultdict(list)
        for original_idx, label in all_data_points:
            class_indices_pool[label].append(original_idx)
        
        for label in class_indices_pool:
            random.shuffle(class_indices_pool[label])

        all_classes = list(class_indices_pool.keys())
        total_classes = len(all_classes)

        if num_classes > total_classes:
            raise ValueError(
                f"Requested {num_classes} classes per client, "
                f"but dataset only has {total_classes} classes."
            )

        rng = np.random.default_rng(seed)
        
        client_turn = 0
        
        while True:
            assigned_this_round = False
            shuffled_classes_for_round = all_classes.copy()
            rng.shuffle(shuffled_classes_for_round)

            for cls in shuffled_classes_for_round:
                if len(class_indices_pool[cls]) > 0:
                    num_to_take = min(samples_per_class_shard, len(class_indices_pool[cls]))
                    
                    if num_to_take > 0:
                        client_id = client_turn % num_clients
                        
                        samples_for_client = class_indices_pool[cls][:num_to_take]
                        class_indices_pool[cls] = class_indices_pool[cls][num_to_take:]
                        
                        client_data[client_id].extend(samples_for_client)
                        assigned_this_round = True
                        
                        client_turn += 1
            
            if not assigned_this_round:
                break

        for i in range(num_clients):
            random.shuffle(client_data[i])

        return dict(client_data)
    '''
    # --- CORRECTED non_iid_split_by_class_dataset ---
    def non_iid_split_by_class_dataset(self, dataset: Dataset, num_clients: int, num_classes_per_client: int = 2, samples_per_class_shard: int = 10, seed: int = 42) -> Dict[int, List[int]]:
        """
        Splits a dataset into non-IID subsets based on class labels for Federated Learning.
        Ensures all clients receive data by distributing 'shards' of classes.

        Args:
            dataset (Dataset): The dataset object (e.g., a PyTorch Dataset or Subset) from which
                               to get labels and indices.
            num_clients (int): The number of clients to split the data among.
            num_classes_per_client (int): The number of distinct classes each client will primarily draw from.
            samples_per_class_shard (int): The number of samples to take from a class each time it's assigned to a client.
                                            Crucial for ensuring data distribution to many clients.
            seed (int): Random seed for reproducibility.

        Returns:
            dict: A dictionary where keys are client IDs (integers)
                  and values are lists of data indices assigned to that client.
                  These indices are relative to the *original dataset* if `dataset` is a Subset,
                  or relative to `dataset` itself if it's a base Dataset.
        """
        np.random.seed(seed) # Ensure reproducibility

        # --- Part 1: Extract labels and map original indices ---
        if isinstance(dataset, Subset):
            base_dataset = dataset.dataset
            subset_indices = dataset.indices
            dataset_labels = np.array([base_dataset.targets[i] for i in subset_indices])
            all_dataset_original_indices = list(subset_indices) # These are the original indices we will assign
        else: # Assume it's a base Dataset (e.g., torchvision.datasets.CIFAR10)
            dataset_labels = np.array(dataset.targets)
            all_dataset_original_indices = list(range(len(dataset))) # These are the original indices

        unique_classes = np.unique(dataset_labels)
        num_total_classes = len(unique_classes)

        if num_classes_per_client > num_total_classes:
            print(f"Warning: num_classes_per_client ({num_classes_per_client}) is greater than "
                  f"total unique classes ({num_total_classes}). Setting to {num_total_classes}.")
            num_classes_per_client = num_total_classes
            
        if samples_per_class_shard <= 0:
            raise ValueError("samples_per_class_shard must be a positive integer.")

        # Create a mutable pool of indices for each class, shuffled
        class_indices_pool = defaultdict(list)
        for idx_in_labels_array, label in enumerate(dataset_labels):
            original_data_index = all_dataset_original_indices[idx_in_labels_array]
            class_indices_pool[label].append(original_data_index)
        
        # Shuffle indices within each class initially
        for label in class_indices_pool:
            random.shuffle(class_indices_pool[label])


        # --- Part 2: Distribute 'shards' of classes to clients ---
        client_data_indices = defaultdict(list)
        rng = np.random.default_rng(seed) # Use modern numpy random generator
        
        # Maintain a list of clients who still need classes assigned to them
        clients_needing_classes = list(range(num_clients))
        rng.shuffle(clients_needing_classes) # Randomize client order

        # Keep track of which class to assign next to ensure rotation
        class_assignment_pointer = 0

        # Loop to assign 'primary' classes to clients
        while clients_needing_classes and any(len(class_indices_pool[cls]) > 0 for cls in unique_classes):
            if class_assignment_pointer >= num_total_classes:
                class_assignment_pointer = 0 # Cycle through classes if needed

            # Take a class to assign
            current_class = unique_classes[class_assignment_pointer]
            
            # If the current class has no samples left, skip it and move to next
            if not class_indices_pool[current_class]:
                class_assignment_pointer += 1
                continue

            # Assign to the next client that still needs classes
            if clients_needing_classes:
                client_id = clients_needing_classes.pop(0) # Get next client
                
                # Assign a shard of the class to this client
                num_to_take = min(samples_per_class_shard * num_classes_per_client, len(class_indices_pool[current_class]))
                if num_to_take > 0:
                    samples_for_client = class_indices_pool[current_class][:num_to_take]
                    class_indices_pool[current_class] = class_indices_pool[current_class][num_to_take:]
                    client_data_indices[client_id].extend(samples_for_client)
                    
                    # Add client back if they still need more distinct classes assigned (up to num_classes_per_client)
                    # This logic is for distributing total classes, not ensuring each client gets num_classes_per_client distinct ones directly
                    # For simplicity, we just distribute all data this way, and then ensure all clients have something.
            
            class_assignment_pointer += 1
            # If all clients have been assigned a class, reset and reshuffle for next round of distribution
            if not clients_needing_classes:
                clients_needing_classes = list(range(num_clients))
                rng.shuffle(clients_needing_classes)


        # --- Part 3: Distribute any remaining data ---
        # This will ensure all remaining samples are distributed to existing clients.
        all_remaining_indices = []
        for cls in unique_classes:
            all_remaining_indices.extend(class_indices_pool[cls])
        
        if all_remaining_indices:
            rng.shuffle(all_remaining_indices)
            for i, idx in enumerate(all_remaining_indices):
                client_id = i % num_clients # Cycle through clients to distribute remaining
                client_data_indices[client_id].append(idx)


        # --- Part 4: Finalize and shuffle client data ---
        # Ensure all clients have at least an empty list if no data was assigned
        for i in range(num_clients):
            if i not in client_data_indices:
                client_data_indices[i] = [] # Ensure all clients have an entry

            # Shuffle the data for each client for randomness
            random.shuffle(client_data_indices[i])

        return dict(client_data_indices)


    