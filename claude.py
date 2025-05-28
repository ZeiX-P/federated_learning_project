import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import timm
import wandb
import numpy as np
from collections import OrderedDict
import copy

class FederatedLearner:
    def __init__(self, num_clients=5, fisher_threshold=0.1, lr=1e-4):
        self.num_clients = num_clients
        self.fisher_threshold = fisher_threshold
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.global_model = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
        # DINO models have Identity head, need to replace with proper classifier
        # Get the feature dimension from the model's embed_dim
        feature_dim = self.global_model.embed_dim  # 384 for vit_small
        self.global_model.head = nn.Linear(feature_dim, 100)  # CIFAR-100
        self.global_model.to(self.device)
        
        # Data preparation
        self.train_loaders, self.test_loader = self._prepare_data()
        
        # Initialize wandb
        wandb.init(project="federated-fisher-dino", 
                  config={"num_clients": num_clients, "fisher_threshold": fisher_threshold, "lr": lr})
        
    def _prepare_data(self):
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        
        # Split training data among clients
        client_size = len(trainset) // self.num_clients
        train_loaders = []
        
        for i in range(self.num_clients):
            start_idx = i * client_size
            end_idx = start_idx + client_size if i < self.num_clients - 1 else len(trainset)
            client_indices = list(range(start_idx, end_idx))
            client_subset = Subset(trainset, client_indices)
            train_loaders.append(DataLoader(client_subset, batch_size=32, shuffle=True))
        
        test_loader = DataLoader(testset, batch_size=128, shuffle=False)
        return train_loaders, test_loader
    
    def compute_fisher_information(self, model, dataloader):
        """Compute Fisher Information Matrix diagonal for model parameters"""
        model.eval()
        fisher_dict = {}
        
        # Initialize Fisher information
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param)
        
        num_samples = 0
        for data, targets in dataloader:
            data, targets = data.to(self.device), targets.to(self.device)
            
            model.zero_grad()
            outputs = model(data)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_dict[name] += param.grad.data ** 2
            
            num_samples += data.size(0)
            if num_samples > 1000:  # Limit samples for efficiency
                break
        
        # Normalize by number of samples
        for name in fisher_dict:
            fisher_dict[name] /= num_samples
            
        return fisher_dict
    
    def get_low_fisher_mask(self, fisher_dict):
        """Create mask for parameters with low Fisher values"""
        mask_dict = {}
        
        for name, fisher_values in fisher_dict.items():
            # Keep parameters with Fisher values below threshold
            mask_dict[name] = fisher_values < self.fisher_threshold
            
        return mask_dict
    
    def local_train(self, client_id, local_epochs=3):
        """Train local model for one client"""
        local_model = copy.deepcopy(self.global_model)
        local_model.train()
        
        optimizer = optim.AdamW(local_model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        # Compute Fisher information
        fisher_dict = self.compute_fisher_information(local_model, self.train_loaders[client_id])
        fisher_mask = self.get_low_fisher_mask(fisher_dict)
        
        total_loss = 0
        num_batches = 0
        
        for epoch in range(local_epochs):
            for data, targets in self.train_loaders[client_id]:
                data, targets = data.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = local_model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                
                # Apply Fisher mask - only update parameters with low Fisher values
                for name, param in local_model.named_parameters():
                    if param.requires_grad and name in fisher_mask:
                        param.grad.data *= fisher_mask[name].float()
                
                optimizer.step()
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Count parameters being updated
        updated_params = sum(mask.sum().item() for mask in fisher_mask.values())
        total_params = sum(param.numel() for param in local_model.parameters() if param.requires_grad)
        
        print(f"Client {client_id}: Loss={avg_loss:.4f}, Updated {updated_params}/{total_params} params ({100*updated_params/total_params:.1f}%)")
        
        return local_model.state_dict(), avg_loss, updated_params/total_params
    
    def federated_averaging(self, client_states):
        """Aggregate client models using federated averaging"""
        global_state = self.global_model.state_dict()
        
        # Initialize aggregated state
        aggregated_state = OrderedDict()
        for key in global_state.keys():
            aggregated_state[key] = torch.zeros_like(global_state[key])
        
        # Average client states
        for client_state in client_states:
            for key in aggregated_state.keys():
                aggregated_state[key] += client_state[key]
        
        for key in aggregated_state.keys():
            aggregated_state[key] /= len(client_states)
        
        return aggregated_state
    
    def evaluate(self):
        """Evaluate global model on test set"""
        self.global_model.eval()
        correct = 0
        total = 0
        test_loss = 0
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.global_model(data)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = test_loss / len(self.test_loader)
        return accuracy, avg_loss
    
    def train(self, num_rounds=20):
        """Main federated training loop"""
        print(f"Starting federated training with {self.num_clients} clients for {num_rounds} rounds")
        
        for round_num in range(num_rounds):
            print(f"\n--- Round {round_num + 1}/{num_rounds} ---")
            
            # Local training
            client_states = []
            client_losses = []
            update_ratios = []
            
            for client_id in range(self.num_clients):
                state, loss, ratio = self.local_train(client_id)
                client_states.append(state)
                client_losses.append(loss)
                update_ratios.append(ratio)
                
                # Log individual client metrics
                wandb.log({
                    "round": round_num + 1,
                    f"client_{client_id}_loss": loss,
                    f"client_{client_id}_update_ratio": ratio,
                    f"client_{client_id}_updated_params": ratio * sum(param.numel() for param in self.global_model.parameters() if param.requires_grad)
                })
            
            # Federated averaging
            aggregated_state = self.federated_averaging(client_states)
            self.global_model.load_state_dict(aggregated_state)
            
            # Evaluation
            accuracy, test_loss = self.evaluate()
            
            # Logging
            avg_client_loss = np.mean(client_losses)
            avg_update_ratio = np.mean(update_ratios)
            
            print(f"Round {round_num + 1}: Test Accuracy={accuracy:.2f}%, Test Loss={test_loss:.4f}")
            print(f"Avg Client Loss={avg_client_loss:.4f}, Avg Update Ratio={avg_update_ratio:.3f}")
            
            # Log aggregated metrics to wandb
            round_metrics = {
                "round": round_num + 1,
                "test_accuracy": accuracy,
                "test_loss": test_loss,
                "avg_client_loss": avg_client_loss,
                "avg_update_ratio": avg_update_ratio,
                "min_client_loss": np.min(client_losses),
                "max_client_loss": np.max(client_losses),
                "std_client_loss": np.std(client_losses),
                "min_update_ratio": np.min(update_ratios),
                "max_update_ratio": np.max(update_ratios)
            }
            
            # Add individual client losses and ratios for comparison
            for client_id in range(self.num_clients):
                round_metrics[f"client_{client_id}_loss_round"] = client_losses[client_id]
                round_metrics[f"client_{client_id}_ratio_round"] = update_ratios[client_id]
            
            wandb.log(round_metrics)
        
        print(f"\nTraining completed! Final accuracy: {accuracy:.2f}%")

# Run federated learning
if __name__ == "__main__":
    fed_learner = FederatedLearner(num_clients=5, fisher_threshold=0.1, lr=1e-4)
    fed_learner.train(num_rounds=15)
    wandb.finish()