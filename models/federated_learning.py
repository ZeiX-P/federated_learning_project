
#from dataset.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch
import wandb
import copy 
from models.train import train_with_global_mask
from typing import Optional, Dict, Union
from torch import nn

from collections import defaultdict
from utils.models_utils import apply_model_diff, compute_model_diff, copy_model
import torch
import copy
import random
import wandb
from torch.utils.data import DataLoader
#from torch.utils.data import Dataset
import logging

class FederatedLearning:

    def __init__(self,
                 global_model: torch.nn.Module,
                 data,
                 num_clients: int,
                 aggregation_method: str,
                 num_rounds: int,
                 epochs_per_round: int,
                 distribution_type: str,
                 client_fraction: float,
                 config,
                 class_per_client,
                 local_steps
            ):

        self.global_model = global_model  
        self.data = data
        self.num_clients = num_clients
        self.aggregation_method = aggregation_method
        self.num_rounds = num_rounds
        self.epochs_per_round = epochs_per_round
        self.distribution_type = distribution_type
        self.client_fraction = client_fraction
        self.config = config
        self.class_per_client = class_per_client
        self.local_steps = local_steps




        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.global_model.to(self.device)
            print("CUDA is available, using GPU.")
        else:
            print("CUDA is not available, using CPU instead.")
            self.device = torch.device('cpu')
            self.global_model.to(self.device)


        

        self.local_models = {i: copy.deepcopy(self.global_model).to(self.device) for i in range(self.num_clients)}
        self.global_train_set, self.global_val_set, self.dict_train_client_data, self.dict_val_client_data = self.d()
        
        # Check client dataset sizes - this helps diagnose data distribution issues
        for client_id in range(self.num_clients):
            train_size = len(self.dict_train_client_data[client_id])
            val_size = len(self.dict_val_client_data[client_id])
            print(f"Client {client_id}: {train_size} training samples, {val_size} validation samples")
            
                # Check class distribution for classification tasks
            try:
                if hasattr(self.dict_train_client_data[client_id].dataset, 'targets'):
                    indices = self.dict_train_client_data[client_id].indices
                    targets = [self.dict_train_client_data[client_id].dataset.targets[i] for i in indices]
                    classes, counts = torch.unique(torch.tensor(targets), return_counts=True)
                    print(f"Client {client_id} class distribution: {dict(zip(classes.tolist(), counts.tolist()))}")
            except Exception as e:
                print(f"Could not analyze class distribution for client {client_id}: {e}")

   
        
    
    def d(self):
        dataset, _ = self.data.get_dataset(self.config.dataset)
        global_train_set, global_val_set = self.data.create_train_val_set(dataset)
        train_indices = self.split_data_to_client(global_train_set, self.num_clients)
        val_indices = self.split_data_to_client(global_val_set, self.num_clients)
        dict_train_client_data = self.create_dict_data(global_train_set, train_indices)
        dict_val_client_data = self.create_dict_data(global_val_set, val_indices)
        return global_train_set, global_val_set, dict_train_client_data, dict_val_client_data
    
    def split_data_to_client(self, dataset, num_clients):
        if self.distribution_type == 'iid':
            indices = self.data.idd_split(dataset, num_clients)
        elif self.distribution_type == 'non-iid-dirichlet':
            indices = self.data.dirichlet_non_iid_split(dataset, num_clients)

        elif self.distribution_type == "non-iid":
            indices = self.data.pathological_non_iid_split(dataset, num_clients,self.class_per_client)
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")
        return indices

    def create_dict_data(self, dataset, indices):
        clients_data = {}
        for client_id, indices_client in indices.items():
            data_client_dataset = torch.utils.data.Subset(dataset, indices_client)
            clients_data[client_id] = data_client_dataset
        return clients_data

    def evaluate_model(self, model, val_loader):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = model(inputs)
                loss = self.config.loss_function(outputs, targets)
                total_loss += loss.item() * targets.size(0)

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        avg_loss = total_loss / total
        accuracy = 100* correct / total
        return avg_loss, accuracy


    def train(self,model, train_loader, val_loader, client, round):

        
        model.train()
        optimizer_params = [p for p in model.parameters() if p.requires_grad]
        #optimizer = torch.optim.SGD(optimizer_params, lr=self.config.learning_rate)

        optimizer = self.config.optimizer_class(
            optimizer_params,
            lr=self.config.learning_rate,
            **self.config.optimizer_params
        )
        #optimizer1 = torch.optim.SGD(optimizer_params, lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler = None
        if self.config.scheduler_class:
            scheduler = self.config.scheduler_class(optimizer, **self.config.scheduler_params)
        #scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=10, gamma=0.1)
        loss_func = self.config.loss_function

        for epoch in range(self.epochs_per_round):
            total_loss = 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.config.loss_function(outputs, targets)
                loss.backward()

                optimizer.step()
                total_loss += loss.item() 
            if scheduler is not None:

                scheduler.step()
                
            # Optional: evaluate on validation set
            val_loss, val_accuracy = self.evaluate_model(model, val_loader)

            # Log training/validation metrics
            wandb.log({
                f"client_{client}/val_loss": val_loss,
                f"client_{client}/val_accuracy": val_accuracy,
                "epoch": epoch,
                "round": round
            })

    
    def train_local_step(self, model, train_loader, val_loader, client, round):
        model.train()
        model.to(self.device)

        optimizer_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = self.config.optimizer_class(
            optimizer_params,
            lr=self.config.learning_rate,
            **self.config.optimizer_params
        )
        optimizer1 = torch.optim.SGD(optimizer_params, lr=0.01, momentum=0.9, weight_decay=1e-4)

        scheduler = None
        if self.config.scheduler_class:
            scheduler = self.config.scheduler_class(optimizer, **self.config.scheduler_params)
        scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=10, gamma=0.1)
        loss_func = self.config.loss_function
        total_loss = 0
        total_samples = 0

        data_iter = iter(train_loader)
        for step in range(self.local_steps):  # <-- local steps instead of epochs
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                inputs, targets = next(data_iter)

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer1.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            optimizer1.step()

            total_loss += loss.item() 
            total_samples += targets.size(0)

            if scheduler1:
                scheduler1.step()

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        val_loss, val_accuracy = self.evaluate_model(model, val_loader)

        
        wandb.log({
            f"client_{client}/train_loss": avg_loss,
            f"client_{client}/val_loss": val_loss,
            f"client_{client}/val_accuracy": val_accuracy,
            "local_step": self.local_steps,
            "round": round
        })


    def aggregate(self,global_model, selected_clients,client_sample_counts):
        
        if self.aggregation_method == 'FedAvg':
            #self.federated_averagingF(global_model, selected_clients)
            self.federated_averaging_aggregate(global_model, selected_clients,client_sample_counts)
        elif self.aggregation_method == 'FedProx':
            self.federated_proximal()
        elif self.aggregation_method == 'FedNova':
            self.federated_nova()
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        wandb.log({"status": "aggregation_complete"})


    def federated_averaging_aggregate(self, 
                                 global_model: torch.nn.Module, 
                                 client_models, 
                                 client_sample_counts) -> torch.nn.Module:
        """
        Implements the FederatedAveraging aggregation as described in the algorithm.
        
        Formula: w_{t+1} = Σ_{k∈S_t} (n_k / m) * w_k^{t+1}
        where:
        - n_k is the number of samples for client k
        - m is the total number of samples across all selected clients
        - w_k^{t+1} is the updated model from client k
        
        Args:
            global_model: The current global model
            client_models: List of updated client models
            client_sample_counts: List of sample counts for each client (n_k values)
        
        Returns:
            Updated global model with aggregated weights
        """
        
        # Calculate total samples across all participating clients
        total_samples = sum(client_sample_counts)
        
        if total_samples == 0:
            raise ValueError("Total samples cannot be zero")
        
        if len(client_models) != len(client_sample_counts):
            raise ValueError("Number of client models must match number of sample counts")
        
        # Get the global model's state dict
        global_dict = global_model.state_dict()
        
        # Initialize aggregated weights to zero
        aggregated_dict = {}
        for key in global_dict:
            aggregated_dict[key] = torch.zeros_like(global_dict[key])
        
        # Aggregate client models weighted by their sample counts
        for client_model, n_k in zip(client_models, client_sample_counts):
            client_dict = client_model.state_dict()
            weight = n_k / total_samples  # This is n_k/m in the formula
            
            for key in aggregated_dict:
                aggregated_dict[key] += weight * client_dict[key].float()
        
        # Load the aggregated weights into the global model
        global_model.load_state_dict(aggregated_dict)
        
        return global_model


    def federated_proximal(self):
        # Basic FedProx implementation with μ=0.01 (proximal term coefficient)
        mu = 0.01
        global_state = self.global_model.state_dict()
        client_weights = [client_model.state_dict() for client_model in self.local_models.values()]
        
        avg_weights = copy.deepcopy(client_weights[0])
        for key in avg_weights:
            avg_weights[key] = torch.zeros_like(avg_weights[key])
            for i in range(len(client_weights)):
                avg_weights[key] += client_weights[i][key]
            avg_weights[key] = avg_weights[key] / len(client_weights)
        
        self.global_model.load_state_dict(avg_weights)
        for client_id in range(self.num_clients):
            self.local_models[client_id].load_state_dict(copy.deepcopy(avg_weights))

    def federated_nova(self):
        # Basic FedNova implementation (simplified)
        tau_eff = self.epochs_per_round  # Effective local steps
        client_weights = [client_model.state_dict() for client_model in self.local_models.values()]
        
        # Normalize updates by effective number of local steps
        avg_weights = copy.deepcopy(client_weights[0])
        for key in avg_weights:
            avg_weights[key] = torch.zeros_like(avg_weights[key])
            for i in range(len(client_weights)):
                avg_weights[key] += client_weights[i][key] / tau_eff
            avg_weights[key] = avg_weights[key] / len(client_weights)
            
        self.global_model.load_state_dict(avg_weights)
        for client_id in range(self.num_clients):
            self.local_models[client_id].load_state_dict(copy.deepcopy(avg_weights))
     


    def validate1(self, model, dataloader):
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # No gradients needed for validation
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = model(inputs)
                loss = self.config.loss_function(outputs, targets)
                val_loss += loss.item() * inputs.size(0)  # Accumulate loss

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        avg_loss = val_loss / total
        accuracy =100* correct / total

        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy
        }

        

    def compute_predictions(self,
    model: nn.Module,
    dataloader: DataLoader
):
        """
        Compute predictions for a given dataloader using the trained model.

        Args:
            model: The trained model.
            dataloader: The DataLoader containing the test or train dataset.
            device: The device to use ('cpu' or 'cuda').
            loss_function: The loss function to minimize in core.

        Returns:
            predictions: Tensor of predictions.
            labels: Tensor of true labels.
            loss: Computed loss for the given data.
            accuracy: Computed accuracy for the given data.
        """
        model.eval()  # Set the model to evaluation mode
        predictions = []
        labels = []
        

        loss = 0.0

        # Disable gradient computation during inference
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(
                    self.device
                )  # Move to the appropriate device

                # Forward pass
                preds = model(inputs)  # Get raw model predictions

                
                loss += self.config.loss_function(preds, targets).item()

                # Get predicted class (class with the highest score)
                _, predicted = torch.max(preds, 1)

                predictions.append(predicted)
                labels.append(targets)

        # Concatenate all predictions and labels
        predictions = torch.cat(predictions)
        labels = torch.cat(labels)

        correct = (predictions == labels).sum().item()
        accuracy = 100.0 * correct / len(labels)
        loss = loss / len(labels)

        return {
            "val_loss": loss,
            "val_accuracy": accuracy,
            "predictions": predictions,
            "labels": labels
        }

        return predictions, labels, loss, accuracy

    def validate(self, model, val_loader):
        """Separate validation function for cleaner code."""
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Handle different data types
                try:
                    # Check if we have model.output_dim attribute
                    output_dim = getattr(model, 'output_dim', None)
                    
                    if isinstance(targets, torch.Tensor) and targets.dtype != torch.long and output_dim == 1:
                        targets = targets.float()
                    elif not isinstance(targets, torch.Tensor):
                        targets = torch.tensor(targets, device=self.device)
                    else:
                        targets = targets.long()
                except Exception as e:
                    print(f"Error in validation target processing: {e}")
                    # Default to making sure targets is a tensor
                    if not isinstance(targets, torch.Tensor):
                        targets = torch.tensor(targets, device=self.device)
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                
                # Adjust output shape if needed
                if outputs.shape != targets.shape and outputs.dim() > 1 and outputs.size(1) == 1:
                    outputs = outputs.squeeze(1)
                
                try:
                    loss = self.config.loss_function(outputs, targets)
                    val_loss += loss.item()
                except Exception as e:
                    logging.error(f"Error calculating loss: {e}")
                 
                    logging.error(f"Output shape: {outputs.shape}, Target shape: {targets.shape}")
                   
                    continue
                
                # Calculate validation accuracy if classification task
                if hasattr(outputs, 'shape') and outputs.dim() > 1 and outputs.size(1) > 1:
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

        # Calculate metrics
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_accuracy = 100 * correct / total if total > 0 else 0
        
        return {
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy
        }

    def evaluate_global_model(self):
        """Evaluate the global model performance on the global validation dataset."""
        # Create a DataLoader for the entire validation set
        val_loader = DataLoader(self.global_val_set, batch_size=self.config.batch_size, shuffle=False)
        
        return self.validate1(self.global_model, val_loader)
    

    def run_federated_learning(self):

        run_name = f"{self.distribution_type},local_steps:{self.local_steps},class_per_client:{self.class_per_client}"
        wandb.init(
            project=self.config.training_name, 
            name=run_name,
            config={
                "aggregation_method": self.aggregation_method,
                "distribution_type": self.distribution_type,
                "num_clients": self.num_clients,
                "num_rounds": self.num_rounds,
                "epochs_per_round": self.epochs_per_round,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "momentum": self.config.momentum,
                "weight_decay": self.config.weight_decay,
                "client_fraction": self.client_fraction,
                "dataset": self.config.dataset,
                "class_per_client": self.class_per_client,
                "local_steps": self.local_steps,
            }
        )
        # Log the model architecture
        wandb.watch(self.global_model)

        for round in range(self.num_rounds):
            print(f"--- Round {round+1}/{self.num_rounds} ---")
            wandb.log({"round_progress": round / self.num_rounds * 100})

            num_selected_clients = max(1, int(self.client_fraction * self.num_clients))
            selected_clients = random.sample(range(self.num_clients), num_selected_clients)

            wandb.log({"active_clients": num_selected_clients, "round": round})

            current_round_trained_models = []
            local_models = []

            client_sample_counts = []

            for client_id in selected_clients:
         
                local_model = copy.deepcopy(self.global_model)
                data_client_train_set = self.dict_train_client_data[client_id]
                data_client_val_set = self.dict_val_client_data[client_id]

              
                client_sample_counts.append(len(data_client_train_set))

                

                train_loader = DataLoader(data_client_train_set, batch_size=self.config.batch_size, shuffle=True)
                val_loader = DataLoader(data_client_val_set, batch_size=self.config.batch_size, shuffle=False)

            
                self.train_local_step(local_model, train_loader, val_loader, client_id, round)
                #self.train(local_model, train_loader, val_loader, client_id, round)

                local_models.append(local_model)

            #self.aggregate(self.global_model, current_round_trained_models)
            self.aggregate(self.global_model, local_models,client_sample_counts)
            global_metrics = self.evaluate_global_model()

            wandb.log({
                "global/val_loss": global_metrics["val_loss"],
                "global/val_accuracy": global_metrics.get("val_accuracy", 0),
                "round": round
            })

            print(f"Round {round+1} - Global validation accuracy: {global_metrics.get('val_accuracy', 0):.2f}%")

    def run_model_editing_federated(self, top_k=0.1):

        run_name = f"{self.distribution_type},local_steps:{self.local_steps},class_per_client:{self.class_per_client},model_editing=YES"
        wandb.init(
            project=self.config.training_name, 
            name=run_name,
            config={
                "aggregation_method": self.aggregation_method,
                "distribution_type": self.distribution_type,
                "num_clients": self.num_clients,
                "num_rounds": self.num_rounds,
                "epochs_per_round": self.epochs_per_round,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "momentum": self.config.momentum,
                "weight_decay": self.config.weight_decay,
                "client_fraction": self.client_fraction,
                "dataset": self.config.dataset,
                "class_per_client": self.class_per_client,
                "local_steps": self.local_steps,
                "model_editing_top_k": top_k 
            }
        )
        # Log the model architecture
        wandb.watch(self.global_model)

        for round in range(self.num_rounds):
            wandb.log({"round": round, "round_progress": (round + 1) / self.num_rounds * 100})

            num_selected_clients = max(1, int(self.client_fraction * self.num_clients))
            selected_clients = random.sample(range(self.num_clients), num_selected_clients)

            wandb.log({
                "active_clients": num_selected_clients,
                "selected_clients": selected_clients
            })
            current_round_trained_models = []
            local_models = []

            client_sample_counts = []


            dict_fisher_scores = {}
            dict_client_masks = {}


            for client_id in selected_clients:
         
                local_model = copy.deepcopy(self.global_model)
                data_client_train_set = self.dict_train_client_data[client_id]
                data_client_val_set = self.dict_val_client_data[client_id]

              
                client_sample_counts.append(len(data_client_train_set))

                

                train_loader = DataLoader(data_client_train_set, batch_size=self.config.batch_size, shuffle=True)
                val_loader = DataLoader(data_client_val_set, batch_size=self.config.batch_size, shuffle=False)

                fisher = self.compute_fisher_information(
                    local_model, train_loader, self.config.loss_function
                )
              

                local_mask = self.generate_mask(fisher, top_k)
                dict_client_masks[client_id] = local_mask
                total_params = sum(m.numel() for m in local_mask.values())
                frozen_params = sum((m == 0).sum().item() for m in local_mask.values())
                sparsity = frozen_params / total_params if total_params > 0 else 0
                wandb.log({
                    f"client_{client_id}/mask_sparsity": sparsity
                })

                self.train_with_global_mask_local_step(local_model, train_loader, val_loader, client_id, round, local_mask)

                local_models.append(local_model)

            # Step 3: Federated aggregation
            self.aggregate(self.global_model, local_models, client_sample_counts)

            # Step 4: Evaluate and log global model
            global_metrics = self.evaluate_global_model()
            wandb.log({
                "global/val_loss": global_metrics["val_loss"],
                "global/val_accuracy": global_metrics.get("val_accuracy", 0)
            })

    def run_centralized_model_editing(self,train_loader,val_loader,top_k):
        """
        Runs the centralized training process with model editing.
        This method mirrors the structure of the federated version for comparison.
        """
        run_name = f"Centralized,lr={self.config.learning_rate},model_editing=YES"
        wandb.init(
            project=self.config.training_name,
            name=run_name,
            config={
                "training_type": "centralized",
                "num_epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "momentum": self.config.momentum,
                "weight_decay": self.config.weight_decay,
                "dataset": self.config.dataset,
                "model_editing_top_k": top_k # Reintroduced for model editing
            }
        )
        #wandb.watch(self.global_model) # Watch the global model (which is self.model)

        print("--- Centralized Training with Model Editing ---")

        # Step 1: Compute Fisher Information and generate mask for the entire dataset
        # This is done ONCE at the beginning for the centralized model
        print("Computing Fisher Information on the full training dataset...")
        fisher_scores = self.compute_fisher_information(
            self.global_model, train_loader, self.config.loss_function
        )
        print("Generating model mask...")
        model_mask = self.generate_mask(fisher_scores, top_k=top_k)
        
        # Log mask sparsity
        total_params = sum(m.numel() for m in model_mask.values())
        frozen_params = sum((m == 0).sum().item() for m in model_mask.values())
        sparsity = frozen_params / total_params if total_params > 0 else 0
        wandb.log({"model/mask_sparsity": sparsity})
        print(f"Mask sparsity: {sparsity:.2f}")

        # Step 2: Apply the binary mask to the global model
        # Parameters with mask[name] == 0 will have requires_grad set to False
        print("Applying mask to model parameters...")
        for name, param in self.global_model.named_parameters():
            if name in model_mask:
                if (model_mask[name] == 0).all():
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(True)
            else: # If a parameter is not in the mask, default to True (trainable)
                param.requires_grad_(True)

        print("Mask applied. Parameters ready for training.")
        # Filter parameters by requires_grad for the optimizer
        optimizer_params = [p for p in self.global_model.parameters() if p.requires_grad]
        if not optimizer_params:
            print("Warning: No parameters are set to require gradients. Model will not train.")
            # Adjust mask generation or raise an error as needed if this state is unexpected

        # Initialize optimizer and scheduler for the masked model
        optimizer = self.config.optimizer_class(
            optimizer_params,
            lr=self.config.learning_rate,
            **self.config.optimizer_params
        )
        optimizer = torch.optim.SGD(optimizer_params, lr=0.001, momentum=0.9, weight_decay=1e-4)
        scheduler = None
        if self.config.scheduler_class:
            scheduler = self.config.scheduler_class(optimizer, **self.config.scheduler_params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        loss_func = self.config.loss_function

        # Step 3: Centralized Training Loop
        for epoch in range(1, self.config.epochs + 1):
            self.global_model.train() # Set model to training mode
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                preds = self.global_model(inputs)
                loss = loss_func(preds, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * targets.size(0)
                _, predicted = preds.max(1)
                total_train += targets.size(0)
                correct_train += predicted.eq(targets).sum().item()

            train_loss = running_loss / total_train
            train_accuracy = 100.0 * correct_train / total_train
      
            if scheduler is not None:
                scheduler.step()

            # Step 4: Evaluate and log metrics
            val_metrics = self.evaluate_global_model()

            wandb.log({
                "Epoch": epoch,
                "global/val_loss": val_metrics["val_loss"],
                "global/val_accuracy": val_metrics.get("val_accuracy", 0)
            })
        wandb.finish()
        print("--- Centralized Training Completed ---")

    def train_with_global_mask_epochs(self, model, train_loader, val_loader, client_id, round, mask):
        # Apply the binary mask using requires_grad
        for name, param in model.named_parameters():
            if name in mask:
                
                if (mask[name] == 0).all(): # This assumes global_mask is binary after generate_global_mask1
                    param.requires_grad_(False)
                else:
                    # If any part of the mask is 1, ensure it's trainable (it might have been frozen before)
                    param.requires_grad_(True)

        

        # Filter parameters by requires_grad
        optimizer_params = [p for p in model.parameters() if p.requires_grad]

        # Initialise local optimizer instance
        optimizer = self.config.optimizer_class(
            optimizer_params,
            lr=self.config.learning_rate,
            **self.config.optimizer_params
        )

        scheduler = None
        if self.config.scheduler_class:
            scheduler = self.config.scheduler_class(optimizer, **self.config.scheduler_params)

        loss_func = self.config.loss_function

        # Training loop
        for epoch in range(1, self.epochs_per_round + 1):
            model.train()
            running_loss = 0.0
            correct, total = 0, 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                preds = model(inputs)
                loss = loss_func(preds, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * targets.size(0)
                _, predicted = preds.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            train_loss = running_loss / total
            train_accuracy = 100.0 * correct / total
      
            if scheduler is not None:
                scheduler.step()

            wandb.log({
                f"client_{client_id}/train_loss": train_loss,
                f"client_{client_id}/train_accuracy": train_accuracy,
                "round": round,
                "epoch": epoch
            })


    def train_with_global_mask_local_step(self, model, train_loader, val_loader, client_id, round, mask):
        # Apply the binary mask using requires_grad
        for name, param in model.named_parameters():
            if name in mask:
                if (mask[name] == 0).all():
                    param.requires_grad_(False)
                else:
                    param.requires_grad_(True)

        # Filter parameters by requires_grad
        optimizer_params = [p for p in model.parameters() if p.requires_grad]
        
        optimizer = self.config.optimizer_class(
            optimizer_params,
            lr=self.config.learning_rate,
            **self.config.optimizer_params
        )
        optimizer = torch.optim.SGD(optimizer_params, lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler = None
        if self.config.scheduler_class:
            scheduler = self.config.scheduler_class(optimizer, **self.config.scheduler_params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        loss_func = self.config.loss_function

        # Local step training loop
        model.train()
        step = 0
        total_steps = self.local_steps  # You must define this in your config or class
        data_iter = iter(train_loader)

        correct, total = 0, 0
        running_loss = 0.0

        while step < total_steps:
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                inputs, targets = next(data_iter)

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            preds = model(inputs)
            loss = loss_func(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            _, predicted = preds.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if scheduler is not None:
                scheduler.step()

            # Optional: log every step
            wandb.log({
                f"client_{client_id}/step_loss": loss.item(),
                "round": round,
                "step": step + 1
            })

            step += 1

        train_loss = running_loss / total
        train_accuracy = 100.0 * correct / total

        # Final log per round
        wandb.log({
            f"client_{client_id}/train_loss": train_loss,
            f"client_{client_id}/train_accuracy": train_accuracy,
            "round": round
        })


    def compute_fisher_diag(self, model, dataloader, loss_fn):

        fisher_diag = None 

        total_samples = 0

        for batch_idx , (inputs, target) in enumerate(dataloader):

            inputs, target = inputs.to(self.device), target.to(self.device)

            model.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, target)
            loss.backward()

            gradients = []

            for parameter in model.parameters():
                if parameter.grad is not None:

                    gradients.append(parameter.grad.detach().clone().flatten())
                else:
                    gradients.append(torch.zeros_like(parameter.data.flatten())) 

            gradients = torch.cat(gradients) 

            if fisher_diag is None:
                fisher_diag = gradients.pow(2)
            else:
                fisher_diag += gradients.pow(2)

            total_samples += 1

        fisher_diag /= total_samples

        return fisher_diag

    def compute_fisher_information(self,model, dataloader, loss_fn, num_samples=1000):
        model.eval()
        fisher = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)

        count = 0
        for inputs, targets in dataloader:
            if count >= num_samples:
                break
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            model.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2

            count += 1

        for name in fisher:
            fisher[name] /= count

        # Log Fisher histograms to wandb
        all_fisher_values = torch.cat([v.flatten().cpu() for v in fisher.values()])
        wandb.log({
            "fisher/all_hist": wandb.Histogram(all_fisher_values)})

        return fisher


        return fisher

    def reshape_fisher_to_named(self, fisher_flat, model):
        param_sizes = [p.numel() for _, p in model.named_parameters()]
        param_shapes = [p.shape for _, p in model.named_parameters()]
        param_names = [name for name, _ in model.named_parameters()]
        split_scores = torch.split(fisher_flat, param_sizes)

        return {
            name: score.view(shape)
            for name, score, shape in zip(param_names, split_scores, param_shapes)
        }

   
    def generate_mask(self,fisher_info, top_k: float = 0.1, strategy: str = "fisher_left_only"):
        if strategy.startswith("fisher"):
            all_scores = torch.cat([f.view(-1) for f in fisher_info.values()])
            
            # Use kthvalue for better memory efficiency with large tensors
            total_elements = all_scores.numel()
            
            if strategy == "fisher_least":
                k = max(1, int(top_k * total_elements))
                threshold = torch.kthvalue(all_scores, k).values
                compare = lambda x: x <= threshold
            elif strategy == "fisher_most":
                k = max(1, int((1 - top_k) * total_elements))
                threshold = torch.kthvalue(all_scores, k).values
                compare = lambda x: x >= threshold
            elif strategy == "fisher_left_only":
                # New strategy: only parameters on the left side of distribution (least important)
                # This sets mask to 1 ONLY for the leftmost top_k fraction of Fisher values
                k = max(1, int(top_k * total_elements))
                threshold = torch.kthvalue(all_scores, k).values
                compare = lambda x: x <= threshold
            else:
                raise ValueError(f"Unknown Fisher strategy: {strategy}")
            
            mask = {name: compare(tensor).float() for name, tensor in fisher_info.items()}

        elif strategy in {"magnitude_lowest", "magnitude_highest"}:
            all_params = torch.cat([p.view(-1).abs() for p in fisher_info.values()])
            total_elements = all_params.numel()
            
            if strategy == "magnitude_lowest":
                k = max(1, int(top_k * total_elements))
                threshold = torch.kthvalue(all_params, k).values
                compare = lambda x: x.abs() <= threshold
            else:
                k = max(1, int((1 - top_k) * total_elements))
                threshold = torch.kthvalue(all_params, k).values
                compare = lambda x: x.abs() >= threshold
            mask = {name: compare(p).float() for name, p in fisher_info.items()}

        elif strategy == "random":
            mask = {
                name: (torch.rand_like(p) < top_k).float()
                for name, p in fisher_info.items()
            }

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return mask
    
    def __del__(self):
        """Clean up when the class is deleted."""
        try:
            # Final evaluation
            final_metrics = self.evaluate_global_model()
            
            # Log final metrics
            wandb.log({
                "final/val_loss": final_metrics["val_loss"],
                "final/val_accuracy": final_metrics["val_accuracy"],
            })
            
            # Save the global model
            torch.save(self.global_model.state_dict(), "final_global_model.pth")
            wandb.save("final_global_model.pth")
            
            # Finish the wandb run
            wandb.finish()
        except:
            # In case of errors during cleanup
            try:
                wandb.finish()
            except:
                pass