import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import timm
import wandb
import copy
import random

# --- Settings ---
NUM_CLIENTS = 5
EPOCHS = 2
ROUNDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.init(project="fedavg-fisher-dino", name="fisher-low-mask")

# --- Load CIFAR-100 ---
transform = T.Compose([T.Resize(224), T.ToTensor()])
trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

client_data = torch.utils.data.random_split(trainset, [len(trainset)//NUM_CLIENTS]*NUM_CLIENTS)

def get_loader(dataset): return torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# --- Fisher Info Computation ---
def compute_fisher(model, dataloader, criterion):
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    model.eval()
    for x, y in dataloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach()**2
    return fisher

# --- Mask: Keep only parameters with lowest Fisher info ---
def mask_low_fisher(fisher, keep_ratio=0.5):
    mask = {}
    for n, f in fisher.items():
        flat = f.view(-1)
        threshold = torch.quantile(flat, keep_ratio)
        mask[n] = (f <= threshold).float()
    return mask

# --- Apply mask during training ---
def train_with_mask(model, dataloader, criterion, mask):
    model.train()
    opt = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(EPOCHS):
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        p.grad *= mask[n].to(DEVICE)
            opt.step()

# --- Evaluation ---
def test(model):
    model.eval()
    loader = get_loader(testset)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    return acc

# --- Federated Training ---
global_model = timm.create_model('vit_small_patch16_224.dino', pretrained=True, num_classes=100).to(DEVICE)
criterion = nn.CrossEntropyLoss()

for rnd in range(ROUNDS):
    local_weights = []

    for cid in range(NUM_CLIENTS):
        local_model = copy.deepcopy(global_model)
        data = get_loader(client_data[cid])
        
        # Fisher info
        fisher = compute_fisher(local_model, data, criterion)
        fisher_mean = torch.stack([f.mean() for f in fisher.values()]).mean().item()
        fisher_std = torch.stack([f.std() for f in fisher.values()]).mean().item()

        # Mask
        mask = mask_low_fisher(fisher, keep_ratio=0.5)
        total, kept = 0, 0
        for m in mask.values():
            total += m.numel()
            kept += m.sum().item()
        sparsity = 1 - (kept / total)

        # Train with mask
        local_model.train()
        opt = optim.Adam(local_model.parameters(), lr=1e-4)
        running_loss, correct_local, total_local = 0, 0, 0

        for epoch in range(EPOCHS):
            for x, y in data:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                out = local_model(x)
                loss = criterion(out, y)
                loss.backward()
                with torch.no_grad():
                    for n, p in local_model.named_parameters():
                        if p.grad is not None:
                            p.grad *= mask[n].to(DEVICE)
                opt.step()
                running_loss += loss.item()
                correct_local += (out.argmax(1) == y).sum().item()
                total_local += y.size(0)

        acc_local = correct_local / total_local
        local_weights.append({n: p.detach().cpu() for n, p in local_model.named_parameters()})

        wandb.log({
            f"round_{rnd}/client_{cid}/fisher_mean": fisher_mean,
            f"round_{rnd}/client_{cid}/fisher_std": fisher_std,
            f"round_{rnd}/client_{cid}/param_sparsity": sparsity,
            f"round_{rnd}/client_{cid}/train_loss": running_loss / len(data),
            f"round_{rnd}/client_{cid}/train_acc": acc_local,
        })

    # Federated averaging
    for name, param in global_model.named_parameters():
        if param.requires_grad:
            param.data = torch.stack([w[name] for w in local_weights], 0).mean(0).to(DEVICE)

    acc = test(global_model)
    wandb.log({f"round_{rnd}/global_test_acc": acc})
    print(f"Round {rnd} - Global Test Acc: {acc:.4f}")
