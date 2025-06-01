import torch
import wandb
import timm
import torch.nn as nn
from config import Configuration
from dataset.data import Dataset
 
from models.train import train_model 

# Sweep configuration
sweep_config = {
    "method": "grid",  # or "grid", "bayes"
    "metric": {
        "name": "Validation Accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "learning_rate": {
            "values": [1e-2, 1e-3, 1e-4]
        },
        "batch_size": {
            "values": [32, 64]
        },
        "momentum": {
            "values": [0.8, 0.9]
        },
        "weight_decay": {
            "values": [1e-4, 5e-4]
        }
    }
}


# Sweep function
def sweep_train():
    with wandb.init() as run:
        config = wandb.config

        # Model setup
        model = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
        for param in model.parameters():
            param.requires_grad = True
        model.head = nn.Linear(384, 100)

        # Create training configuration
        training_config = Configuration(
            model=model,
            training_name=f"sweep_run_{run.name}",
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            dataset="CIFAR100",
            optimizer_class=torch.optim.SGD,
            loss_function=nn.CrossEntropyLoss(),
            scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
            epochs=10,
            project_name="fl_centralized_baseline",
            optimizer_params={
                "momentum": config.momentum,
                "weight_decay": config.weight_decay
            },
            scheduler_params={"T_max": 20}
        )

        # Data
        data = Dataset()
        train_loader, val_loader = data.get_dataloaders(training_config.dataset)

        # Train
        train_model(
            training_params=training_config,
            train_loader=train_loader,
            val_loader=val_loader,
            project_name="fl_centralized_baseline",
        )


if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="fl_centralized_baseline")
    wandb.agent(sweep_id, function=sweep_train, count=10)  # Adjust count as needed