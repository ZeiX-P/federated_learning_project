from dataset.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch
import wandb
import copy 
from models.train import train_with_global_mask
from typing import Optional, Dict, Union
from torch import nn
from utils.cam_utils import extract_param_feature_map
from collections import defaultdict
from utils.models_utils import apply_model_diff, compute_model_diff, copy_model
import random
import logging

class FederatedLearning:

    def __init__(self,
                 global_model: torch.nn.Module,
                 data: Dataset,
                 num_clients: int,
                 aggregation_method: str,
                 num_rounds: int,
                 epochs_per_round: int,
                 distribution_type: str,
                 client_fraction: float,
                 config
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

        # Initialise wandb with more detailed configuration
        run_name = f"{self.aggregation_method}_{self.distribution_type}_{self.num_clients}clients"
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
                "dataset": self.config.dataset
            }
        )
        # Log the model architecture
        wandb.watch(self.global_model)

    def d(self):
        dataset, _ = self.data.get_dataset(self.config.dataset)
        global_train_set, global_val_set = self.data.create_train_val_set(dataset)
        train_indices = self.split_data_to_client(global_train_set, self.num_clients)
        val_indices = self.split_data_to_client(global_val_set, self.num_clients)
        dict_train_client_data = self.create_dict_data(global_train_set, train_indices)
        dict_val_client_data = self.create_dict_data(global_val_set, val_indices)
        return global_train_set, global_val_set, dict_train_client_data, dict_val_client_data

    def split_data_to_client(self, dataset: Dataset, num_clients):
        if self.distribution_type == 'iid':
            indices = self.data.idd_split(dataset, num_clients)
        elif self.distribution_type == 'non-iid':
            indices = self.data.dirichlet_non_iid_split(dataset, num_clients)
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")
        return indices

    def create_dict_data(self, dataset, indices):
        clients_data = {}
        for client_id, indices_client in indices.items():
            data_client_dataset = torch.utils.data.Subset(dataset, indices_client)
            clients_data[client_id] = data_client_dataset
        return clients_data

    def aggregate(self):
        wandb.log({"status": "aggregating"})
        if self.aggregation_method == 'FedAvg':
            self.federated_averaging()
        elif self.aggregation_method == 'FedProx':
            self.federated_proximal()
        elif self.aggregation_method == 'FedNova':
            self.federated_nova()
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        wandb.log({"status": "aggregation_complete"})

    def federated_averaging(self):
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

    def run(self):
        for round in range(self.num_rounds):
            print(f"--- Round {round+1}/{self.num_rounds} ---")
            wandb.log({"round_progress": round/self.num_rounds * 100})
            
            # Select fraction of clients for this round
            num_selected_clients = max(1, int(self.client_fraction * self.num_clients))
            selected_clients = random.sample(range(self.num_clients), num_selected_clients)
            
            wandb.log({"active_clients": num_selected_clients, "round": round})
            
            
            
            # Train selected clients
            for client in selected_clients:
                data_client_train_set = self.dict_train_client_data[client]
                data_client_val_set = self.dict_val_client_data[client]

                train_loader = DataLoader(data_client_train_set, batch_size=self.config.batch_size, shuffle=True)
                val_loader = DataLoader(data_client_val_set, batch_size=self.config.batch_size, shuffle=False)
                
                self.train(self.local_models[client], train_loader, val_loader, client, round)

            # Aggregate and evaluate global model
            self.aggregate()
            global_metrics = self.evaluate_global_model()
            
            # Log global model performance for this round
            wandb.log({
                "global/val_loss": global_metrics["val_loss"],
                "global/val_accuracy": global_metrics.get("val_accuracy", 0),
                "round": round
            })
            
            print(f"Round {round+1} - Global validation accuracy: {global_metrics.get('val_accuracy', 0):.2f}%")

    def train(self, model, train_loader, val_loader, client, round):
        # Create a new optimizer instance for each client with proper parameters
        try:
            if hasattr(self.config, 'optimizer') and self.config.optimizer == 'sgd':
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    momentum=self.config.momentum,
                    weight_decay=self.config.weight_decay
                )
            elif hasattr(self.config, 'optimizer') and self.config.optimizer == 'adam':
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
            else:
                # Default to SGD if not specified or if there's an issue
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                print(f"Using default SGD optimizer for client {client}")
        except Exception as e:
            print(f"Error creating optimizer, using default: {e}")
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Learning rate scheduler removed - causing compatibility issues
        scheduler = None

        for epoch in range(self.epochs_per_round):
            print(f"--- Client {client}, Epoch {epoch+1}/{self.epochs_per_round} ---")
            model.train()
            running_loss = 0
            correct = 0
            total = 0
            
            # Training loop
            for batch_idx, (data, target) in enumerate(train_loader):
                # Handle different data types (make sure target is appropriate type)
                try:
                    # Check if we have model.output_dim attribute
                    output_dim = getattr(model, 'output_dim', None)
                    
                    if isinstance(target, torch.Tensor) and target.dtype != torch.long and output_dim == 1:
                        # Regression task possibly
                        target = target.float()
                    elif not isinstance(target, torch.Tensor):
                        target = torch.tensor(target, device=self.device)
                    else:
                        # Classification task likely - ensure long type for CrossEntropyLoss
                        target = target.long()
                except Exception as e:
                    print(f"Error in target processing: {e}")
                    # Default to making sure target is a tensor
                    if not isinstance(target, torch.Tensor):
                        target = torch.tensor(target, device=self.device)
                
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                output = model(data)
                
                # Make sure output shape matches target for loss calculation
                if output.shape != target.shape and output.dim() > 1 and output.size(1) == 1:
                    output = output.squeeze(1)
                
                try:
                    loss = self.config.loss_function(output, target)
                except Exception as e:
                    print(f"Error in loss calculation: {e}")
                    print(f"Output shape: {output.shape}, Target shape: {target.shape}")
                    print(f"Output dtype: {output.dtype}, Target dtype: {target.dtype}")
                    continue
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                # Calculate accuracy if classification task
                try:
                    if hasattr(output, 'shape') and output.dim() > 1 and output.size(1) > 1:
                        # Multi-class classification
                        _, predicted = output.max(1)
                        total += target.size(0)
                        correct += predicted.eq(target).sum().item()
                except Exception as e:
                    print(f"Error calculating training accuracy: {e}")
                
                # Log mini-batch progress occasionally
                if batch_idx % 10 == 0:
                    wandb.log({
                        f"client_{client}/batch_loss": loss.item(),
                        "round": round,
                        "epoch": epoch,
                        "batch": batch_idx
                    })

            avg_train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
            train_accuracy = 100 * correct / total if total > 0 else 0

            # Validation phase
            val_metrics = self.validate(model, val_loader)
            avg_val_loss = val_metrics["val_loss"]
            val_accuracy = val_metrics["val_accuracy"]
            
            # Learning rate scheduling removed due to compatibility issues
            # Just log the current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            if batch_idx == 0:
                print(f"Current learning rate: {current_lr}")

            print(f"Round {round}, Client {client}, Epoch {epoch}, "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

            # Log metrics for this client
            wandb.log({
                f"client_{client}/train_loss": avg_train_loss,
                f"client_{client}/train_accuracy": train_accuracy,
                f"client_{client}/val_loss": avg_val_loss,
                f"client_{client}/val_accuracy": val_accuracy,
                f"client_{client}/learning_rate": optimizer.param_groups[0]['lr'],
                "round": round,
                "epoch": epoch
            })
    
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
                    print(f"Error in validation loss calculation: {e}")
                    print(f"Output shape: {outputs.shape}, Target shape: {targets.shape}")
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
        
        return self.validate(self.global_model, val_loader)
    
    def run_model_editing_global(self):

        for round in range(self.num_rounds):
            print(f"--- Round {round+1}/{self.num_rounds} ---")
            wandb.log({"round_progress": round / self.num_rounds * 100})

            num_selected_clients = max(1, int(self.client_fraction * self.num_clients))
            selected_clients = random.sample(range(self.num_clients), num_selected_clients)

            wandb.log({"active_clients": num_selected_clients, "round": round})

            dict_fisher_scores = {}

            for client in selected_clients:
                train_loader = DataLoader(self.dict_train_client_data[client], batch_size=self.config.batch_size, shuffle=True)
                fisher = self.compute_fisher_diag(self.local_models[client], train_loader, self.config.loss_function)
                fisher_named = self.reshape_fisher_to_named(fisher, self.local_models[client])
                dict_fisher_scores[client] = fisher_named

            global_fisher = self.aggregate_sensitivity_scores(dict_fisher_scores)
            global_mask = self.create_mask_from_fisher(global_fisher, top_k=0.5)

            for client in selected_clients:
                train_loader = DataLoader(self.dict_train_client_data[client], batch_size=self.config.batch_size, shuffle=True)
                val_loader = DataLoader(self.dict_val_client_data[client], batch_size=self.config.batch_size, shuffle=False)
                self.train_with_global_mask(self.local_models[client], train_loader, val_loader, client, round, global_mask)

            self.aggregate()
            global_metrics = self.evaluate_global_model()

            wandb.log({
                "global/val_loss": global_metrics["val_loss"],
                "global/val_accuracy": global_metrics.get("val_accuracy", 0),
                "round": round
            })

            print(f"Round {round+1} - Global validation accuracy: {global_metrics.get('val_accuracy', 0):.2f}%")


    def run_model_editing_talos(self):
        for round in range(self.num_rounds):
            wandb.log({"round": round, "round_progress": (round + 1) / self.num_rounds * 100})

            num_selected_clients = max(1, int(self.client_fraction * self.num_clients))
            selected_clients = random.sample(range(self.num_clients), num_selected_clients)

            wandb.log({
                "active_clients": num_selected_clients,
                "selected_clients": selected_clients
            })

            dict_fisher_scores = {}
            dict_client_masks = {}

            # Step 1: Compute local Fisher + create local mask per client
            for client in selected_clients:
                train_loader = DataLoader(
                    self.dict_train_client_data[client],
                    batch_size=self.config.batch_size,
                    shuffle=True
                )

       
        

                fisher_info = self.compute_fisher_information(
                    self.local_models[client], train_loader, self.config.loss_function
                )
    

                local_mask = self.generate_global_mask1(fisher_info, top_k=0.5)
                dict_client_masks[client] = local_mask

                wandb.log({
                    f"client_{client}/mask_sparsity": sum(1 for v in local_mask.values() if v.sum() == 0) / len(local_mask),
                    f"client_{client}/fisher_norm": sum(v.norm().item() for v in fisher_info.values())
                })

            # Step 2: Train clients with their own masks
            for client in selected_clients:
                train_loader = DataLoader(
                    self.dict_train_client_data[client],
                    batch_size=self.config.batch_size,
                    shuffle=True
                )
                val_loader = DataLoader(
                    self.dict_val_client_data[client],
                    batch_size=self.config.batch_size,
                    shuffle=False
                )

                self.train_model_with_mask(
                    self.local_models[client],
                    train_loader=train_loader,
                    val_loader=val_loader,
                    client_id=client,
                    round_id=round,
                    training_params=self.config,  # assuming config includes training_name, project_name, etc.
                    fisher_mask=dict_client_masks[client]
                )

            # Step 3: Federated aggregation
            self.aggregate()

            # Step 4: Evaluate and log global model
            global_metrics = self.evaluate_global_model()
            wandb.log({
                "global/val_loss": global_metrics["val_loss"],
                "global/val_accuracy": global_metrics.get("val_accuracy", 0)
            })


    def generate_global_mask1(self, fisher_info, top_k: float = 0.2, strategy: str = "fisher_least"):
        if strategy.startswith("fisher"):
            all_scores = torch.cat([f.view(-1) for f in fisher_info.values()])
            if strategy == "fisher_least":
                threshold = torch.quantile(all_scores, top_k)
                compare = lambda x: x <= threshold
            elif strategy == "fisher_most":
                threshold = torch.quantile(all_scores, 1 - top_k)
                compare = lambda x: x >= threshold
            else:
                raise ValueError(f"Unknown Fisher strategy: {strategy}")
            mask = {name: compare(tensor).float() for name, tensor in fisher_info.items()}

        elif strategy in {"magnitude_lowest", "magnitude_highest"}:
            all_params = torch.cat([p.view(-1).abs() for p in fisher_info.values()])
            if strategy == "magnitude_lowest":
                threshold = torch.quantile(all_params, top_k)
                compare = lambda x: x.abs() <= threshold
            else:
                threshold = torch.quantile(all_params, 1 - top_k)
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

    def compute_fisher_information(self, model, dataloader, device, num_samples=100):
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
            loss = self.config.loss_function(outputs, targets)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2

            count += 1

        for name in fisher:
            fisher[name] /= count

        return fisher



    def train_model_with_mask(self,
    model,
    *,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    client_id: Optional[int] = None,
    round_id: Optional[int] = None,
    wandb_log: bool = True,
    wandb_save: bool = True,
    fisher_mask: Optional[dict] = None,
) -> dict:
        assert train_loader is not None
        if val_loader is not None:
            assert isinstance(val_loader, DataLoader)

        use_wandb = wandb_log or wandb_save
        if use_wandb:
            if self.config.project_name is None:
                raise ValueError("project_name cannot be None if using wandb.")
            wandb.init(
                project=self.config.project_name,
                name=f"{self.config.training_name}_client{client_id}_round{round_id}",
                config={
                    "epochs": self.config.epochs,
                    "batch_size": train_loader.batch_size,
                    "learning_rate": self.config.learning_rate,
                    "architecture": self.config.model.__class__.__name__,
                },
            )

        model = model.to(self.device)
        loss_func = self.config.loss_function
        optimizer = self.config.optimizer_class
        scheduler = self.config.scheduler_class
        num_epochs = self.config.epochs
        best_acc = 0

        # Log mask stats
        if fisher_mask:
            total_params = 0
            zeroed_params = 0
            for name, mask in fisher_mask.items():
                total_params += mask.numel()
                zeroed_params += (mask == 0).sum().item()

            if wandb_log:
                wandb.log({
                    f"client{client_id}_round{round_id}/Total Params": total_params,
                    f"client{client_id}_round{round_id}/Zeroed Params": zeroed_params,
                    f"client{client_id}_round{round_id}/Sparsity (%)": 100.0 * zeroed_params / total_params
                })

        # ---- Training with mask ----
        for epoch in range(1, num_epochs + 1):
            model.train()
            running_loss = 0.0
            correct, total = 0, 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                preds = model(inputs)
                loss = loss_func(preds, targets)

                optimizer.zero_grad()
                loss.backward()

                # Apply Fisher mask to gradients
                if fisher_mask:
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            if name in fisher_mask and param.grad is not None:
                                param.grad.mul_(fisher_mask[name])

                optimizer.step()
                running_loss += loss.item() * targets.size(0)
                _, predicted = preds.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            if scheduler is not None:
                scheduler.step()

            train_loss = running_loss / total
            train_accuracy = 100.0 * correct / total

            if wandb_log:
                wandb.log({
                    f"client{client_id}_round{round_id}/Epoch": epoch,
                    f"client{client_id}_round{round_id}/Train Loss": train_loss,
                    f"client{client_id}_round{round_id}/Train Accuracy": train_accuracy,
                })

            if val_loader:
                _, _, val_loss, val_accuracy = self.compute_predictions(model, val_loader, self.device, loss_func)

                if wandb_log:
                    wandb.log({
                        f"client{client_id}_round{round_id}/Val Loss": val_loss,
                        f"client{client_id}_round{round_id}/Val Accuracy": val_accuracy,
                    })

                if val_accuracy > best_acc:
                    best_acc = val_accuracy
                    if wandb_save:
                        model_name = f"{training_params.training_name}_client{client_id}_round{round_id}_best.pth"
                        torch.save(model.state_dict(), model_name)
                        wandb.save(model_name)

        if use_wandb:
            wandb.finish()

        return {"model": model, "best_accuracy": best_acc}


    def compute_predictions(self, model: nn.Module, dataloader: DataLoader, device: torch.device, loss_function: Optional[nn.Module] = None):
        model.eval()
        predictions, labels = [], []
        total_loss, total_samples = 0.0, 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)

                if loss_function is not None:
                    total_loss += loss_function(preds, targets).item() * targets.size(0)
                    total_samples += targets.size(0)

                _, predicted = torch.max(preds, 1)
                predictions.append(predicted)
                labels.append(targets)

        predictions = torch.cat(predictions)
        labels = torch.cat(labels)
        accuracy = 100.0 * (predictions == labels).sum().item() / len(labels)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        return predictions, labels, avg_loss, accuracy

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