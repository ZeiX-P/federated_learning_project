
from collections import defaultdict
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import logging
from typing import List, Tuple, Optional, Dict

import numpy as np
import random

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
            
            # Distribute indices round-robin style for perfect balance
            for idx, data_idx in enumerate(indices):
                client_id = idx % num_clients
                indices_clients[client_id].append(data_idx)

            self._validate_iid_split(dataset, indices_clients, num_clients)
            
            return indices_clients
        
    def _validate_iid_split(self, dataset, indices_clients, num_clients):
        """Validate that the IID split maintains class distribution"""
        
        logging.info("Starting IID split validation")
        
        # Get labels for the dataset
        if hasattr(dataset, 'targets'):
            labels = dataset.targets
            logging.debug("Using dataset.targets for labels")
        elif hasattr(dataset, 'labels'):
            labels = dataset.labels
            logging.debug("Using dataset.labels for labels")
        else:
            # For custom datasets, you might need to extract labels differently
            labels = [dataset[i][1] for i in range(len(dataset))]
            logging.debug("Extracting labels from dataset items")
        
        labels = np.array(labels)
        
        # Overall class distribution
        unique_classes, overall_counts = np.unique(labels, return_counts=True)
        overall_dist = overall_counts / len(labels)
        
        class_dist_str = ", ".join([f"Class {cls}: {dist:.3f}" 
                                for cls, dist in zip(unique_classes, overall_dist)])
        logging.info(f"Overall class distribution: {class_dist_str}")
        
        # Check each client's distribution
        max_deviation = 0
        deviations = []
        
        for client_id in range(num_clients):
            client_labels = labels[indices_clients[client_id]]
            client_unique, client_counts = np.unique(client_labels, return_counts=True)
            client_dist = client_counts / len(client_labels)
            
            # Calculate deviation from overall distribution
            deviation = 0
            for cls in unique_classes:
                if len(overall_dist) > 0:
                    overall_prop = overall_dist[unique_classes == cls][0]
                else:
                    overall_prop = 0
                client_prop = client_dist[client_unique == cls][0] if cls in client_unique else 0
                deviation += abs(overall_prop - client_prop)
            
            deviations.append(deviation)
            max_deviation = max(max_deviation, deviation)
            
            # Log detailed info for first few clients
            if client_id < 3:
                client_dist_str = ", ".join([f"Class {cls}: {dist:.3f}" 
                                        for cls, dist in zip(client_unique, client_dist)])
                logging.info(f"Client {client_id}: {len(client_labels)} samples, "
                        f"distribution: [{client_dist_str}], deviation: {deviation:.4f}")
        
        # Summary statistics
        mean_deviation = np.mean(deviations)
        std_deviation = np.std(deviations)
        
        logging.info(f"Deviation statistics - Max: {max_deviation:.4f}, "
                f"Mean: {mean_deviation:.4f}, Std: {std_deviation:.4f}")
        
        # Log first 5 client sizes for verification
        first_5_sizes = [len(indices_clients[i]) for i in range(min(5, num_clients))]
        logging.info(f"First 5 client data sizes: {first_5_sizes}")
        
        # Quality assessment
        if max_deviation < 0.1:
            logging.info("✓ IID split quality: EXCELLENT (max deviation < 0.1)")
        elif max_deviation < 0.2:
            logging.warning("⚠ IID split quality: GOOD (max deviation < 0.2)")
        else:
            logging.error(f"✗ IID split quality: POOR (max deviation = {max_deviation:.4f})")
        
        logging.info("IID split validation completed")

   
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


    