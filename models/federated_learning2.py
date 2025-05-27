import time
import random
from typing import Dict, List, Optional, Tuple, Any

# Scientific computing and data handling
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # Optional: for better plot styling

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

# Weights & Biases for experiment tracking
import wandb

# Optional: For enhanced plotting and statistics
from scipy import stats  # For statistical analysis
import pandas as pd      # For data manipulation if needed

# Optional: For progress bars (if you want visual progress in terminal)
from tqdm import tqdm



def run_model_editing_talos(self):
        # Initialize comprehensive tracking
        round_metrics = []
        client_performance = {client: [] for client in range(self.num_clients)}
        
        for round in range(self.num_rounds):
            round_start_time = time.time()
            
            # Basic round logging
            wandb.log({
                "round": round, 
                "round_progress": (round + 1) / self.num_rounds * 100,
                "timestamp": time.time()
            })

            num_selected_clients = max(1, int(self.client_fraction * self.num_clients))
            selected_clients = random.sample(range(self.num_clients), num_selected_clients)

            wandb.log({
                "federation/active_clients": num_selected_clients,
                "federation/total_clients": self.num_clients,
                "federation/client_fraction": self.client_fraction,
                "federation/selected_clients": selected_clients
            })

            dict_client_masks = {}
            round_fisher_stats = []
            round_sparsity_stats = []

            # Step 1: Compute local Fisher + create local mask per client
            print(f"Round {round}: Computing Fisher Information and Masks...")
            
            for client in selected_clients:
                client_start_time = time.time()
                
                train_loader = DataLoader(
                    self.dict_train_client_data[client],
                    batch_size=self.config.batch_size,
                    shuffle=True
                )

                fisher_info = self.compute_fisher_information(
                    self.local_models[client], train_loader, self.device, num_samples=100
                )

                local_mask = self.generate_global_mask1(fisher_info, top_k=0.1, strategy="fisher_least")
                dict_client_masks[client] = local_mask

                # Enhanced Fisher and mask statistics
                fisher_norm = sum(v.norm().item() for v in fisher_info.values())
                fisher_mean = sum(v.mean().item() for v in fisher_info.values()) / len(fisher_info)
                fisher_std = sum(v.std().item() for v in fisher_info.values()) / len(fisher_info)
                
                mask_sparsity = sum(1 for v in local_mask.values() if v.sum() == 0) / len(local_mask)
                total_params = sum(v.numel() for v in local_mask.values())
                masked_params = sum((v == 0).sum().item() for v in local_mask.values())
                sparsity_percentage = 100.0 * masked_params / total_params
                
                round_fisher_stats.append(fisher_norm)
                round_sparsity_stats.append(sparsity_percentage)

                client_compute_time = time.time() - client_start_time

                wandb.log({
                    f"clients/client_{client}/fisher_norm": fisher_norm,
                    f"clients/client_{client}/fisher_mean": fisher_mean,
                    f"clients/client_{client}/fisher_std": fisher_std,
                    f"clients/client_{client}/mask_sparsity": mask_sparsity,
                    f"clients/client_{client}/sparsity_percentage": sparsity_percentage,
                    f"clients/client_{client}/total_params": total_params,
                    f"clients/client_{client}/masked_params": masked_params,
                    f"clients/client_{client}/compute_time": client_compute_time,
                    "round": round
                })

            # Aggregate Fisher statistics across clients
            wandb.log({
                "fisher_stats/mean_fisher_norm": np.mean(round_fisher_stats),
                "fisher_stats/std_fisher_norm": np.std(round_fisher_stats),
                "fisher_stats/min_fisher_norm": np.min(round_fisher_stats),
                "fisher_stats/max_fisher_norm": np.max(round_fisher_stats),
                "sparsity_stats/mean_sparsity": np.mean(round_sparsity_stats),
                "sparsity_stats/std_sparsity": np.std(round_sparsity_stats),
                "sparsity_stats/min_sparsity": np.min(round_sparsity_stats),
                "sparsity_stats/max_sparsity": np.max(round_sparsity_stats),
                "round": round
            })

            # Step 2: Train clients with their own masks
            print(f"Round {round}: Training clients with masks...")
            
            client_train_results = {}
            training_times = []
            
            for client in selected_clients:
                client_train_start = time.time()
                
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

                result = self.train_model_with_mask(
                    self.local_models[client],
                    train_loader=train_loader,
                    val_loader=val_loader,
                    client_id=client,
                    round_id=round,
                    fisher_mask=dict_client_masks[client]
                )
                
                client_train_time = time.time() - client_train_start
                training_times.append(client_train_time)
                client_train_results[client] = result
                
                # Store client performance for trend analysis
                client_performance[client].append({
                    'round': round,
                    'accuracy': result["best_accuracy"],
                    'training_time': client_train_time
                })

                wandb.log({
                    f"clients/client_{client}/final_accuracy": result["best_accuracy"],
                    f"clients/client_{client}/training_time": client_train_time,
                    "round": round
                })

            # Training time statistics
            wandb.log({
                "training_stats/mean_training_time": np.mean(training_times),
                "training_stats/std_training_time": np.std(training_times),
                "training_stats/total_training_time": sum(training_times),
                "round": round
            })

            # Step 3: Federated aggregation
            aggregation_start = time.time()
            self.aggregate()
            aggregation_time = time.time() - aggregation_start
            
            wandb.log({
                "federation/aggregation_time": aggregation_time,
                "round": round
            })

            # Step 4: Evaluate and log global model
            eval_start = time.time()
            global_metrics = self.evaluate_global_model()
            eval_time = time.time() - eval_start
            
            round_total_time = time.time() - round_start_time
            
            # Enhanced global metrics logging
            wandb.log({
                "global/val_loss": global_metrics["val_loss"],
                "global/val_accuracy": global_metrics.get("val_accuracy", 0),
                "global/evaluation_time": eval_time,
                "timing/round_total_time": round_total_time,
                "timing/round_efficiency": global_metrics.get("val_accuracy", 0) / round_total_time,
                "round": round
            })
            
            # Store round metrics for trend analysis
            round_metrics.append({
                'round': round,
                'global_accuracy': global_metrics.get("val_accuracy", 0),
                'global_loss': global_metrics["val_loss"],
                'mean_client_accuracy': np.mean([result["best_accuracy"] for result in client_train_results.values()]),
                'round_time': round_total_time
            })
            
            # Client diversity metrics
            client_accuracies = [result["best_accuracy"] for result in client_train_results.values()]
            wandb.log({
                "diversity/client_accuracy_std": np.std(client_accuracies),
                "diversity/client_accuracy_range": np.max(client_accuracies) - np.min(client_accuracies),
                "diversity/client_accuracy_cv": np.std(client_accuracies) / np.mean(client_accuracies) if np.mean(client_accuracies) > 0 else 0,
                "round": round
            })
            
            # Log progress and ETA
            if round > 0:
                avg_round_time = np.mean([m['round_time'] for m in round_metrics])
                eta_seconds = avg_round_time * (self.num_rounds - round - 1)
                wandb.log({
                    "progress/eta_minutes": eta_seconds / 60,
                    "progress/completion_percentage": (round + 1) / self.num_rounds * 100,
                    "round": round
                })
            
            # Create custom plots every few rounds
            if (round + 1) % max(1, self.num_rounds // 10) == 0 or round == self.num_rounds - 1:
                self._create_custom_plots(round_metrics, client_performance, round)

        # Final summary statistics
        self._log_final_summary(round_metrics, client_performance)

    def _create_custom_plots(self, round_metrics, client_performance, current_round):
        """Create custom plots for WandB"""
        
        # 1. Global model performance over rounds
        rounds = [m['round'] for m in round_metrics]
        global_accs = [m['global_accuracy'] for m in round_metrics]
        global_losses = [m['global_loss'] for m in round_metrics]
        
        # Accuracy trend plot
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, global_accs, 'b-', linewidth=2, label='Global Accuracy')
        plt.xlabel('Round')
        plt.ylabel('Accuracy (%)')
        plt.title('Global Model Performance Over Rounds')
        plt.grid(True, alpha=0.3)
        plt.legend()
        wandb.log({f"plots/global_accuracy_trend": wandb.Image(plt)})
        plt.close()
        
        # Loss trend plot
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, global_losses, 'r-', linewidth=2, label='Global Loss')
        plt.xlabel('Round')
        plt.ylabel('Loss')
        plt.title('Global Model Loss Over Rounds')
        plt.grid(True, alpha=0.3)
        plt.legend()
        wandb.log({f"plots/global_loss_trend": wandb.Image(plt)})
        plt.close()
        
        # 2. Client performance comparison
        if len(client_performance) > 1:
            plt.figure(figsize=(12, 8))
            for client_id, performance in client_performance.items():
                if performance:  # Only plot if client has data
                    client_rounds = [p['round'] for p in performance]
                    client_accs = [p['accuracy'] for p in performance]
                    plt.plot(client_rounds, client_accs, '-o', label=f'Client {client_id}', alpha=0.7)
            
            plt.xlabel('Round')
            plt.ylabel('Accuracy (%)')
            plt.title('Client Performance Comparison')
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            wandb.log({f"plots/client_performance_comparison": wandb.Image(plt)})
            plt.close()
        
        # 3. Round time analysis
        round_times = [m['round_time'] for m in round_metrics]
        plt.figure(figsize=(10, 6))
        plt.plot(rounds, round_times, 'g-', linewidth=2, label='Round Time')
        plt.axhline(y=np.mean(round_times), color='orange', linestyle='--', label=f'Average ({np.mean(round_times):.2f}s)')
        plt.xlabel('Round')
        plt.ylabel('Time (seconds)')
        plt.title('Round Execution Time')
        plt.grid(True, alpha=0.3)
        plt.legend()
        wandb.log({f"plots/round_time_analysis": wandb.Image(plt)})
        plt.close()

    def _log_final_summary(self, round_metrics, client_performance):
        """Log final summary statistics"""
        
        final_global_acc = round_metrics[-1]['global_accuracy'] if round_metrics else 0
        final_global_loss = round_metrics[-1]['global_loss'] if round_metrics else float('inf')
        
        # Convergence analysis
        if len(round_metrics) > 1:
            acc_improvement = final_global_acc - round_metrics[0]['global_accuracy']
            loss_improvement = round_metrics[0]['global_loss'] - final_global_loss
            
            wandb.log({
                "summary/final_global_accuracy": final_global_acc,
                "summary/final_global_loss": final_global_loss,
                "summary/accuracy_improvement": acc_improvement,
                "summary/loss_improvement": loss_improvement,
                "summary/total_rounds": len(round_metrics),
                "summary/total_training_time": sum(m['round_time'] for m in round_metrics)
            })
        
        # Client summary statistics
        all_client_final_accs = []
        for client_id, performance in client_performance.items():
            if performance:
                final_acc = performance[-1]['accuracy']
                all_client_final_accs.append(final_acc)
                
                wandb.log({
                    f"summary/client_{client_id}_final_accuracy": final_acc,
                    f"summary/client_{client_id}_total_rounds": len(performance)
                })
        
        if all_client_final_accs:
            wandb.log({
                "summary/mean_client_final_accuracy": np.mean(all_client_final_accs),
                "summary/std_client_final_accuracy": np.std(all_client_final_accs),
                "summary/min_client_final_accuracy": np.min(all_client_final_accs),
                "summary/max_client_final_accuracy": np.max(all_client_final_accs)
            })

    def generate_global_mask1(self, fisher_info, top_k: float = 0.2, strategy: str = "fisher_least"):
        mask = {}
        mask_stats = {'total_params': 0, 'masked_params': 0}
        
        for name, tensor in fisher_info.items():
            flat_tensor = tensor.view(-1)
            if strategy == "fisher_most":
                threshold = torch.quantile(flat_tensor, 1 - top_k)
                mask_tensor = (tensor >= threshold).float()
            elif strategy == "fisher_least":
                threshold = torch.quantile(flat_tensor, top_k)
                mask_tensor = (tensor <= threshold).float()
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            mask[name] = mask_tensor
            mask_stats['total_params'] += tensor.numel()
            mask_stats['masked_params'] += (mask_tensor == 0).sum().item()
        
        return mask

    def compute_fisher_information(self, model, dataloader, device, num_samples=100):
        model.eval()
        fisher = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param)

        count = 0
        computation_times = []
        
        for inputs, targets in dataloader:
            if count >= num_samples:
                break
                
            batch_start = time.time()
            inputs, targets = inputs.to(device), targets.to(device)
            model.zero_grad()
            outputs = model(inputs)
            loss = self.config.loss_function(outputs, targets)
            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2

            count += 1
            computation_times.append(time.time() - batch_start)

        for name in fisher:
            fisher[name] /= count

        # Log Fisher computation statistics
        wandb.log({
            "fisher_computation/samples_processed": count,
            "fisher_computation/avg_batch_time": np.mean(computation_times),
            "fisher_computation/total_time": sum(computation_times)
        })

        return fisher

    def train_model_with_mask(self, model, *, train_loader, val_loader=None, client_id=None, round_id=None, wandb_log=True, wandb_save=True, fisher_mask=None):
        model = model.to(self.device)
        loss_func = self.config.loss_function
        optimizer = self.config.optimizer_class(model.parameters(), lr=self.config.learning_rate)

        scheduler = None
        if self.config.scheduler_class is not None:
            scheduler = self.config.scheduler_class(optimizer, T_max=self.config.epochs)

        best_acc = 0
        best_loss = float('inf')
        epoch_metrics = []

        # Enhanced mask statistics
        if fisher_mask:
            total_params = 0
            zeroed_params = 0
            layer_sparsity = {}
            
            for name, mask in fisher_mask.items():
                layer_params = mask.numel()
                layer_zeroed = (mask == 0).sum().item()
                total_params += layer_params
                zeroed_params += layer_zeroed
                layer_sparsity[name] = 100.0 * layer_zeroed / layer_params

            overall_sparsity = 100.0 * zeroed_params / total_params

            if wandb_log:
                wandb.log({
                    f"training/client_{client_id}_round_{round_id}/total_params": total_params,
                    f"training/client_{client_id}_round_{round_id}/zeroed_params": zeroed_params,
                    f"training/client_{client_id}_round_{round_id}/overall_sparsity": overall_sparsity,
                    "round": round_id
                })
                
                # Log per-layer sparsity
                for layer_name, sparsity in layer_sparsity.items():
                    wandb.log({
                        f"training/client_{client_id}_round_{round_id}/layer_sparsity/{layer_name}": sparsity,
                        "round": round_id
                    })

        for epoch in range(1, self.config.epochs + 1):
            epoch_start_time = time.time()
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            batch_losses = []

            for batch_idx, (inputs, targets) in enumerate(train_loader):
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
                                if param.grad.shape == fisher_mask[name].shape:
                                    param.grad.mul_(fisher_mask[name])

                optimizer.step()
                
                batch_loss = loss.item()
                running_loss += batch_loss * targets.size(0)
                batch_losses.append(batch_loss)
                
                _, predicted = preds.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            if scheduler is not None:
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step()
                new_lr = optimizer.param_groups[0]['lr']
                
                if wandb_log:
                    wandb.log({
                        f"training/client_{client_id}_round_{round_id}/learning_rate": new_lr,
                        f"training/client_{client_id}_round_{round_id}/lr_change": new_lr - old_lr,
                        "epoch": epoch,
                        "round": round_id
                    })

            epoch_time = time.time() - epoch_start_time
            train_loss = running_loss / total
            train_accuracy = 100.0 * correct / total

            # Enhanced epoch logging
            if wandb_log:
                wandb.log({
                    f"training/client_{client_id}_round_{round_id}/epoch": epoch,
                    f"training/client_{client_id}_round_{round_id}/train_loss": train_loss,
                    f"training/client_{client_id}_round_{round_id}/train_accuracy": train_accuracy,
                    f"training/client_{client_id}_round_{round_id}/epoch_time": epoch_time,
                    f"training/client_{client_id}_round_{round_id}/batch_loss_std": np.std(batch_losses),
                    f"training/client_{client_id}_round_{round_id}/batch_loss_mean": np.mean(batch_losses),
                    "round": round_id
                })

            # Validation evaluation
            val_loss, val_accuracy = None, None
            if val_loader:
                _, _, val_loss, val_accuracy = self.compute_predictions(model, val_loader, self.device, loss_func)
                if wandb_log:
                    wandb.log({
                        f"training/client_{client_id}_round_{round_id}/val_loss": val_loss,
                        f"training/client_{client_id}_round_{round_id}/val_accuracy": val_accuracy,
                        f"training/client_{client_id}_round_{round_id}/train_val_gap": train_accuracy - val_accuracy,
                        "epoch": epoch,
                        "round": round_id
                    })

                # Track best model
                if val_accuracy > best_acc:
                    best_acc = val_accuracy
                    best_loss = val_loss
                    if wandb_save:
                        model_name = f"{self.config.training_name}_client{client_id}_round{round_id}_best.pth"
                        torch.save(model.state_dict(), model_name)
                        wandb.save(model_name)
            
            # Store epoch metrics for analysis
            epoch_metrics.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'epoch_time': epoch_time
            })

        # Final training summary
        total_training_time = sum(m['epoch_time'] for m in epoch_metrics)
        if wandb_log:
            wandb.log({
                f"training/client_{client_id}_round_{round_id}/total_training_time": total_training_time,
                f"training/client_{client_id}_round_{round_id}/avg_epoch_time": total_training_time / len(epoch_metrics),
                f"training/client_{client_id}_round_{round_id}/best_val_accuracy": best_acc,
                f"training/client_{client_id}_round_{round_id}/best_val_loss": best_loss,
                f"training/client_{client_id}_round_{round_id}/final_train_accuracy": epoch_metrics[-1]['train_accuracy'],
                "round": round_id
            })

        return {"model": model, "best_accuracy": best_acc, "best_loss": best_loss, "epoch_metrics": epoch_metrics}

    def compute_predictions(self, model, dataloader, device, loss_function=None):
        model.eval()
        predictions, labels = [], []
        total_loss, total_samples = 0.0, 0
        inference_times = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                batch_start = time.time()
                inputs, targets = inputs.to(device), targets.to(device)
                preds = model(inputs)
                
                if loss_function is not None:
                    total_loss += loss_function(preds, targets).item() * targets.size(0)
                    total_samples += targets.size(0)
                    
                _, predicted = torch.max(preds, 1)
                predictions.append(predicted)
                labels.append(targets)
                inference_times.append(time.time() - batch_start)

        predictions = torch.cat(predictions)
        labels = torch.cat(labels)
        accuracy = 100.0 * (predictions == labels).sum().item() / len(labels)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

        # Log inference statistics
        wandb.log({
            "inference/avg_batch_time": np.mean(inference_times),
            "inference/total_samples": len(labels),
            "inference/throughput_samples_per_sec": len(labels) / sum(inference_times)
        })

        return predictions, labels, avg_loss, accuracy