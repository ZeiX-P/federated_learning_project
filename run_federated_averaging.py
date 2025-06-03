
from config import Configuration
from dataset.data import Dataset
from models.federated_learning import FederatedLearning
import timm
import torch
import torch.nn as nn


if __name__ == "__main__":

    data = Dataset()
    '''
    config = Configuration(
                          batch_size=32,
                          learning_rate=0.01,
                          momentum=0.9,
                          weight_decay=5e-4,
                          dataset="CIFAR100",
                          optimizer=torch.optim.SGD,
                          loss_function=nn.CrossEntropyLoss())
    
    '''
    
    dino = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
    #for param in dino.parameters():
     #   param.requires_grad = False
    dino.head = nn.Linear(384, 100)

    config1 = Configuration(
                          model = dino,
                          training_name="federated_learning_model_editing_talos",
                          batch_size=64,
                          learning_rate=1e-4,
                          momentum=0.9,
                          weight_decay=5e-4,
                          dataset="CIFAR100",
                          optimizer_class=torch.optim.SGD,
                          loss_function=nn.CrossEntropyLoss(),
                          scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
                          epochs=10,
                          optimizer_params={"momentum": 0.9, "weight_decay": 5e-4},
                          scheduler_params={"T_max": 20},
                          project_name="federated_learning_model_editing_talos")

    federated_learning = FederatedLearning(global_model=dino,data=data, num_clients=10, 
                                           aggregation_method="FedAvg", num_rounds=10,
                                            epochs_per_round=4, distribution_type="iid",
                                            client_fraction=0.1,config=config1)
    
    print("Starting Federated Learning process...")
    federated_learning.run_federated_learning()

    print("Federated Learning process completed.")