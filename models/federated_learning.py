from config import Configuration
from dataset.data import Dataset
from torch.utils.data import DataLoader
import torch

import copy 

class FederatedLearning:

    def __init__(self,
                 global_model: torch.nn.Module,
                 data: Dataset,
                 num_clients: int,
                 aggregation_method: str,
                 num_rounds: int,
                 epochs_per_round: int,
                 distribution_type: str,
                 config: Configuration,
            ):

        self.global_model = global_model  
        self.data = data
        self.num_clients = num_clients
        self.aggregation_method = aggregation_method
        self.num_rounds = num_rounds
        self.epochs_per_round = epochs_per_round
        self.distribution_type = distribution_type
        self.config = config

        indices = self.split_data_to_client()
        self.clients_data = self.create_dict_data(indices)

        self.local_models = {i: copy.deepcopy(self.global_model) for i in range(self.num_clients)}


    def split_data_to_client(self):

        if self.distribution_type == 'iid':
            # IID data distribution
            indices = self.data.idd_split(self.data.train_set, self.num_clients)
        if self.distribution_type == 'non-iid':
            # Non-IID data distribution
            indices = self.data.nidd_split(self.data.train_set, self.num_clients)
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")

        return indices
            
    def create_dict_data(self, indices):

        clients_data = {}

        for client_id, indices_client in indices.items():
            data_client_train_set = torch.utils.data.Subset(self.data.train_set, indices_client)
            data_client_val_set = torch.utils.data.Subset(self.data.val_set, indices_client)
            clients_data[client_id] = (data_client_train_set, data_client_val_set)

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

    def federated_proximal(self):
        pass 

    def federated_nova(self):
        pass
     
    def run(self):

        for round in range(self.num_rounds):
            print(f"--- Round {round+1}/{self.num_rounds} ---")

            for client in range(self.num_clients):

                data_client_train_set = self.clients_data[client][0]
                data_client_val_set = self.clients_data[client][1]

                train_loader = DataLoader(data_client_train_set, batch_size=self.config.batch_size, shuffle=True)
                val_loader = DataLoader(data_client_val_set, batch_size=self.config.batch_size, shuffle=False)
                
                self.train(self.local_models[client], train_loader, val_loader, self.config.loss_function, self.config.optimizer, client)               

            self.aggregate()

    def train(self,model, train_loader, val_loader, loss_function, optimizer, client):

        for epoch in range(self.epochs_per_round):
            model.train()
            for batch in train_loader:
                data, target = batch
                optimizer.zero_grad()
                output = model(data)
                loss = loss_function(output, target)
                loss.backward()
                optimizer.step()

            # Validation step
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in val_loader:
                    data, target = batch
                    output = model(data)
                    val_loss += loss_function(output, target).item()

            print(f"Round {round}, Client {client}, Epoch {epoch}, Validation Loss: {val_loss / len(val_loader)}")
            


