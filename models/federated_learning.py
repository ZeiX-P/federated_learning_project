#from config import Configuration
from dataset.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch
import wandb
import copy 
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