#from config import Configuration
from dataset.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch
import wandb
import copy 
from models.train import train_with_global_mask
from typing import Optional, Dict, Union
from torch import nn

'''
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
                 config: Configuration
            ):

        self.global_model = global_model  
        self.data = data
        self.num_clients = num_clients
        self.aggregation_method = aggregation_method
        self.num_rounds = num_rounds
        self.epochs_per_round = epochs_per_round
        self.distribution_type = distribution_type
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

        self.dict_train_client_data, self.dict_val_client_data = self.d()

    def d(self):

        dataset, _ = self.data.get_dataset(self.config.dataset)

        global_train_set, global_val_set = self.data.create_train_val_set(dataset)
        train_indices = self.split_data_to_client(global_train_set, self.num_clients)
        val_indices = self.split_data_to_client(global_val_set, self.num_clients)

        dict_train_client_data = self.create_dict_data(global_train_set,train_indices)
        dict_val_client_data = self.create_dict_data(global_val_set,val_indices)

        return dict_train_client_data, dict_val_client_data

    def split_data_to_client(self, dataset: Dataset, num_clients):

        if self.distribution_type == 'iid':
            # IID data distribution
            indices = self.data.idd_split(dataset, num_clients)
        elif self.distribution_type == 'non-iid':
            # Non-IID data distribution
            indices = self.data.dirichlet_non_iid_split(dataset, num_clients)
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")

        return indices
            
    def create_dict_data(self,dataset, indices):

        clients_data = {}

        for client_id, indices_client in indices.items():
            data_client_dataset = torch.utils.data.Subset(dataset, indices_client)
            
            clients_data[client_id] = data_client_dataset

        return clients_data

    def aggregate(self):

        if self.aggregation_method == 'FedAvg':
            # Federated Averaging
            self.federated_averaging()
        elif self.aggregation_method == 'FedProx':
            # Federated Proximal
            self.federated_proximal()
        elif self.aggregation_method == 'FedNova':
            # Federated Nova
            self.federated_nova()
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def federated_averaging(self):

        client_weights = [client_model.state_dict() for client_model in self.client_models]

        avg_weights = copy.deepcopy(client_weights[0])

        for key in avg_weights:
            for i in range(1, len(client_weights)):
                avg_weights[key] += client_weights[i][key]
            avg_weights[key] = avg_weights[key] / len(client_weights)  

        self.global_model.load_state_dict(avg_weights)

        for client_id in range(self.num_clients):
                self.local_models[client_id].load_state_dict(copy.deepcopy(avg_weights))

    def federated_proximal(self):
        pass 

    def federated_nova(self):
        pass
     
    def run(self):

        for round in range(self.num_rounds):
            print(f"--- Round {round+1}/{self.num_rounds} ---")

            for client in range(self.num_clients):

                data_client_train_set = self.dict_train_client_data[client]
                data_client_val_set = self.dict_val_client_data[client]

                train_loader = DataLoader(data_client_train_set, batch_size=self.config.batch_size, shuffle=True)
                val_loader = DataLoader(data_client_val_set, batch_size=self.config.batch_size, shuffle=False)
                
                self.train(self.local_models[client], train_loader, val_loader, client)               

            self.aggregate()

            

    def train(self,model, train_loader, val_loader, client):
        
        optimizer = self.config.optimizer(model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        
        for epoch in range(self.epochs_per_round):
            print(f"--- Client {client}, Epoch {epoch+1}/{self.epochs_per_round} ---")
            model.train()
            print(len(train_loader))
            for data, target in train_loader:
                print("i")
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = model(data)
                loss = self.config.loss_function(output, target)
                loss.backward()
                optimizer.step()

            # Validation step
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for inputs, targets in val_loader:
                    print("k")
                    inputs, targets = inputs.to(self.device), targets.to(self.device)  
                    output = model(inputs)
                    val_loss += self.config.loss_function(output, targets).item()

            print(f"Round {round}, Client {client}, Epoch {epoch}, Validation Loss: {val_loss / len(val_loader)}")
    
    
'''



import torch
import copy
import random
import wandb
from torch.utils.data import DataLoader, Dataset
'''
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
        self.global_train_set, self.global_val_set,self.dict_train_client_data, self.dict_val_client_data = self.d()

        # Initialise wandb with more detailed configuration
        run_name = f"{self.aggregation_method}_{self.distribution_type}_{self.num_clients}clients"
        wandb.init(
            project="federated_learning_project", 
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
            for i in range(1, len(client_weights)):
                avg_weights[key] += client_weights[i][key]
            avg_weights[key] = avg_weights[key] / len(client_weights)
        self.global_model.load_state_dict(avg_weights)
        for client_id in range(self.num_clients):
            self.local_models[client_id].load_state_dict(copy.deepcopy(avg_weights))

    def federated_proximal(self):
        pass

    def federated_nova(self):
        pass

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

    def train(self, model, train_loader, val_loader, client, round):
        optimizer = self.config.optimizer


        for epoch in range(self.epochs_per_round):
            print(f"--- Client {client}, Epoch {epoch+1}/{self.epochs_per_round} ---")
            model.train()
            running_loss = 0
            correct = 0
            total = 0
            
            # Training loop
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = self.config.loss_function(output, target)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                # Calculate accuracy if classification task
                if hasattr(output, 'argmax'):
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()
                
                # Log mini-batch progress occasionally
                if batch_idx % 10 == 0:
                    wandb.log({
                        f"client_{client}/batch_loss": loss.item(),
                        "round": round,
                        "epoch": epoch,
                        "batch": batch_idx
                    })

            avg_train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct / total if total > 0 else 0

            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    output = model(inputs)
                    val_loss += self.config.loss_function(output, targets).item()
                    
                    # Calculate validation accuracy if classification task
                    if hasattr(output, 'argmax'):
                        _, predicted = output.max(1)
                        val_total += targets.size(0)
                        val_correct += predicted.eq(targets).sum().item()

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0

            print(f"Round {round}, Client {client}, Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

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

    def evaluate_global_model(self):
        """Evaluate the global model performance on all validation data."""
        # Create a combined validation dataset
        combined_val_data = []
        for client_id in range(self.num_clients):
            combined_val_data.extend(self.dict_val_client_data[client_id])
        
        if not combined_val_data:
            return {"val_loss": 0, "val_accuracy": 0}
            
        val_loader = DataLoader(combined_val_data, batch_size=self.config.batch_size, shuffle=False)
        
        self.global_model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.global_model(inputs)
                loss = self.config.loss_function(outputs, targets)
                val_loss += loss.item()
                
                # Calculate accuracy if classification task
                if hasattr(outputs, 'argmax'):
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
                
'''

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
            project="federated_learning_project", 
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
    
    def run_model_editing(self):

        for round in range(self.num_rounds):
            print(f"--- Round {round+1}/{self.num_rounds} ---")
            wandb.log({"round_progress": round/self.num_rounds * 100})
            
            # Select fraction of clients for this round
            num_selected_clients = max(1, int(self.client_fraction * self.num_clients))
            selected_clients = random.sample(range(self.num_clients), num_selected_clients)
            
            wandb.log({"active_clients": num_selected_clients, "round": round})
            
            dict_local_mask = {}

            for client in selected_clients:
                data_client_train_set = self.dict_train_client_data[client]

                train_loader = DataLoader(data_client_train_set, batch_size=self.config.batch_size, shuffle=True)
                
                fisher = self.compute_fisher_diag(self.local_models[client], train_loader,self.config.loss_function)
                mask = self.create_fisher_mask(fisher, self.local_models[client], 0.5)
                dict_local_mask[client] = mask

            global_mask = self.aggregate_sensitivity_scores(dict_local_mask, 0.5)
            # Train selected clients
            for client in selected_clients:
                data_client_train_set = self.dict_train_client_data[client]
                data_client_val_set = self.dict_val_client_data[client]

                train_loader = DataLoader(data_client_train_set, batch_size=self.config.batch_size, shuffle=True)
                val_loader = DataLoader(data_client_val_set, batch_size=self.config.batch_size, shuffle=False)
                
                self.train_with_global_mask(self.local_models[client], train_loader, val_loader, client, round)

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


    def create_fisher_mask(self,fisher_diagonal: torch.Tensor, sparsity_ratio: float, model: nn.Module) -> Dict[str, torch.Tensor]:
        if fisher_diagonal is None:
            wandb.log({"error": "Fisher diagonal tensor is None"})
            raise ValueError("Fisher diagonal tensor is None")
        # Calculate the number of parameters to freeze.
        num_params = fisher_diagonal.numel()
        num_freeze = int(sparsity_ratio * num_params)

        # Handle the case where sparsity_ratio is 0.
        if num_freeze == 0:
            return {name: torch.ones_like(param) for name, param in model.named_parameters()}

        # Find the threshold value.
        threshold_value = torch.kthvalue(fisher_diagonal, num_freeze).values

        # Create the mask.
        flat_mask = (fisher_diagonal < threshold_value).float()  # 1 for parameters to update, 0 for frozen

        param_sizes = [p.numel() for _, p in model.named_parameters()]
        param_shapes = [p.shape for _, p in model.named_parameters()]
        param_names = [name for name, _ in model.named_parameters()]
        split_masks = torch.split(flat_mask, param_sizes)

        return {
            name: mask.view(shape)
            for name, mask, shape in zip(param_names, split_masks, param_shapes)
        }
    
    def aggregate_sensitivity_scores(self, dict_local_scores, threshold=0.5):

        total_data_points = sum(len(self.dict_train_client_data[client]) for client in dict_local_scores)
        aggregated_scores = {}

        for client, scores in dict_local_scores.items():
            client_data_size = len(self.dict_train_client_data[client])
            weight = client_data_size / total_data_points

            for name, score in scores.items():
                if name not in aggregated_scores:
                    aggregated_scores[name] = score.clone().float() * weight
                else:
                    aggregated_scores[name] += score.float() * weight

        # Apply threshold to get the final global binary mask
        global_mask = {
            name: (score >= threshold).int()
            for name, score in aggregated_scores.items()
        }

        return global_mask

    def train_with_global_mask(self, model, train_loader, val_loader, client_id, round):

        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.config.lr, momentum=0.9)
        loss_fn = self.config.loss_function
        global_mask = self.global_mask  # Assume this is set somewhere before training

        for epoch in range(self.config.epochs):
            total_loss = 0
            correct = 0
            total = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()

                # Apply global mask: zero gradients for frozen parameters
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None and name in global_mask:
                            param.grad *= global_mask[name]

                optimizer.step()

                total_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            avg_loss = total_loss / total
            accuracy = 100. * correct / total

            wandb.log({
                f"client_{client_id}/train_loss": avg_loss,
                f"client_{client_id}/train_accuracy": accuracy,
                "epoch": epoch,
                "round": round
            })

        # Optional: Evaluate on validation set
        val_loss, val_accuracy = self.evaluate_model(model, val_loader)
        wandb.log({
            f"client_{client_id}/val_loss": val_loss,
            f"client_{client_id}/val_accuracy": val_accuracy,
            "round": round
        })


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