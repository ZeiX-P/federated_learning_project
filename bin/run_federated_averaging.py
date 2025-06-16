
from config import Configuration
from dataset.data import Dataset
from models.federated_learning import FederatedLearning
import timm
import torch
import torch.nn as nn
import logging

if __name__ == "__main__":

    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

    data = Dataset()
    
    dino = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
    for param in dino.parameters():
        param.requires_grad = False
    dino.head = nn.Linear(384, 100)

    for param in dino.head.parameters():
        param.requires_grad = True

    config1 = Configuration(
                          model = dino,
                          training_name="FLM",
                          batch_size=64,
                          learning_rate=0.000001,
                          momentum=0.9,
                          weight_decay=5e-4,
                          dataset="CIFAR100",
                          optimizer_class=torch.optim.SGD,
                          loss_function=nn.CrossEntropyLoss(),
                          scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
                          epochs=10,
                          optimizer_params={"momentum": 0.9, "weight_decay": 5e-4},
                          scheduler_params={"T_max": 20},
                          project_name="FLM")

    federated_learning = FederatedLearning(global_model=dino,data=data, num_clients=100, 
                                           aggregation_method="FedAvg", num_rounds=100,
                                            epochs_per_round=4, distribution_type="non-iid",
                                            client_fraction=0.1,config=config1, class_per_client=10,local_steps=8)
    
    print("Starting Federated Learning process...")
    federated_learning.run_federated_learning()

    print("Federated Learning process completed.")