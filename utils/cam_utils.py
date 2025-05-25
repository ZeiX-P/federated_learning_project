from torchcam.methods import GradCAM

def extract_param_feature_map(model, dataloader, device, target_layer="layer4", num_samples=3):
    model.eval()
    model.to(device)

    cam_extractor = GradCAM(model, target_layer=target_layer)
    feature_map = {}

    samples = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        for i in range(images.size(0)):
            if samples >= num_samples:
                break
            heatmap = cam_extractor(preds[i].item(), outputs[i].unsqueeze(0))[0]

            for name, param in model.named_parameters():
                if param.requires_grad and param.ndim >= 2:
                    feature_map.setdefault(name, []).append(heatmap.mean().item())
            samples += 1
        if samples >= num_samples:
            break

    return {k: sum(v)/len(v) for k, v in feature_map.items()}