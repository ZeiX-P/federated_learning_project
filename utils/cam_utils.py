from torchcam.methods import GradCAM

import torch
import torch.nn as nn
from torchcam.methods import GradCAM # Make sure GradCAM is imported
# from torchcam.utils import overlay_mask # You might need this for visualization, not directly for the fix

def extract_param_feature_map(model, dataloader, device, target_layer, num_samples=6):
    model.eval()
    model.to(device)

    # Initialize GradCAM extractor
    # Note: For timm Vision Transformers, 'target_layer' might need to be a specific module object,
    # e.g., model.blocks[-1].norm1, not just a string "layer4".
    # Ensure 'target_layer' here matches the actual module you want to attribute.
    # The previous error "TypeError: 'VisionTransformer' object is not subscriptable" was fixed by
    # passing the actual module: temp_model_for_gradcam.blocks[-1].norm1
    cam_extractor = GradCAM(model, target_layer=target_layer)
    feature_map = {}

    samples = 0
    # It's good practice to wrap inference parts in torch.no_grad()
    # However, GradCAM requires gradients for its backward pass,
    # so we will rely on GradCAM managing that internally.
    # If the model itself has layers that require gradients for GradCAM,
    # ensure they are set to requires_grad=True *before* this function is called,
    # as discussed in previous steps with the temporary model.

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Ensure that the model is in eval mode here as it was set at the start
        # and that the input to the model will build a graph for backprop for CAM.
        # This means, if you're using a `with torch.no_grad():` block for the whole inference,
        # it might need to be removed for this forward pass.
        # However, GradCAM typically re-enables necessary gradients internally.
        outputs = model(images)
        preds = outputs.argmax(dim=1)

        for i in range(images.size(0)):
            if samples >= num_samples:
                break # Stop if enough samples are processed

            # --- FIX IS HERE ---
            # Pass retain_graph=True to allow for multiple backward passes if needed
            # from the same computational graph (e.g., iterating through a batch).
            heatmap = cam_extractor(preds[i].item(), outputs[i].unsqueeze(0), retain_graph=True)[0]
            # -------------------

            # Your logic to extract feature map based on heatmap
            for name, param in model.named_parameters():
                if param.requires_grad and param.ndim >= 2:
                    # Append mean of heatmap to list for each parameter
                    # This implies 'feature_map' will store a list of mean heatmap values per parameter
                    feature_map.setdefault(name, []).append(heatmap.mean().item())
            samples += 1

        # Stop processing batches if enough samples are processed
        if samples >= num_samples:
            break

    # Aggregate the collected heatmap means for each parameter
    # e.g., by averaging them across samples
    # This will return a dictionary where each key is a parameter name
    # and the value is the average of the mean heatmaps collected for it.
    final_feature_signature = {k: sum(v) / len(v) for k, v in feature_map.items()}

    model.train() # Set model back to train mode when done with feature extraction

    return final_feature_signature