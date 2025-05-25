import torch.nn as nn
import timm

dino = timm.create_model('vit_small_patch16_224.dino', pretrained=True)

print("Model's named modules:")
for name, module in dino.named_modules():
    print(name)