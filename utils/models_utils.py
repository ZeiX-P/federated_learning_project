

def apply_model_diff(self, global_model, update_dict):
    for name, param in global_model.named_parameters():
        if name in update_dict and param.requires_grad:
            param.data += update_dict[name].to(param.device)

def compute_model_diff(self, model_before, model_after):
    diff = {}
    for (name, p_before), (_, p_after) in zip(model_before.named_parameters(), model_after.named_parameters()):
        if p_before.requires_grad:
            diff[name] = p_after.data.clone() - p_before.data.clone()
    return diff
