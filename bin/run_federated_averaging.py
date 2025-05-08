from models.federated_learning import FederatedLearning
from config import Configuration
from dataset.data import Dataset
import timm
import torch
import torch.nn as nn


if __name__ == "__main__":

    data = Dataset()
    config = Configuration(
                          batch_size=32,
                          learning_rate=0.01,
                          momentum=0.9,
                          weight_decay=5e-4,
                          dataset="CIFAR100",
                          optimizer=torch.optim.SGD,
                          loss_function=nn.CrossEntropyLoss())
    
    dino = timm.create_model('vit_small_patch16_224.dino', pretrained=True)

    federated_learning = FederatedLearning(global_model=dino,data=data, num_clients=10, 
                                           aggregation_method="FedAvg", num_rounds=10,
                                            epochs_per_round=3, distribution_type="iid",
                                            client_fraction=0.5,config=config)
    
    print("Starting Federated Learning process...")
    federated_learning.run()

    print("Federated Learning process completed.")