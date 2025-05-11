from models.federated_learning import FederatedLearning
from config import Configuration
from dataset.data import Dataset
import timm
import torch
import torch.nn as nn
from models.train import train_model 

if __name__ == "__main__":

    data = Dataset()
    dino = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
    
    config = Configuration(
                          model = dino,
                          training_name="fl_centralized_baseline",
                          batch_size=64,
                          learning_rate=0.01,
                          momentum=0.9,
                          weight_decay=5e-4,
                          dataset="CIFAR100",
                          optimizer_class=torch.optim.SGD,
                          loss_function=nn.CrossEntropyLoss(),
                          scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
                          epochs=10,
                          optimizer_params={"momentum": 0.9, "weight_decay": 5e-4},
                          scheduler_params={"T_max": 20})
    
    
    
    train_dataloader, val_dataloader = data.get_dataloaders(config.dataset)

    res_dict = train_model(
        training_params=config,
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        project_name="fl_centralized_baseline",
    )