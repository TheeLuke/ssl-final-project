import os
import torch
import math
import numpy as np

def clip_gradients(model, clip):
    """
    Clips gradients of the model parameters.
    Crucial for ViT stability to prevent exploding gradients.
    """
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms

def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    """
    In DINO, we freeze the last layer (prototypes) for the first epoch
    to allow the backbone to stabilize before we start clustering.
    """
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None

def restart_from_checkpoint(ckpt_path, run_variables=None, **kwargs):
    """
    Re-loads weights from a checkpoint if it exists.
    kwargs: models and optimizers to load.
    """
    if not os.path.isfile(ckpt_path):
        return
    print(f"Found checkpoint at {ckpt_path}")
    
    # Map location ensures we don't run out of memory if loading from a different device setup
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(f"=> Loaded '{key}' from checkpoint with msg: {msg}")
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print(f"=> Loaded '{key}' from checkpoint")
                except ValueError:
                    print(f"=> Failed to load '{key}'")
        else:
            print(f"=> Key '{key}' not found in checkpoint")

    # Reload the epoch counter
    if run_variables is not None:
        if 'epoch' in checkpoint:
            run_variables['epoch'] = checkpoint['epoch']

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    """
    Creates a cosine schedule for learning rates or weight decay.
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(math.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def save_checkpoint(path, state):
    """
    Saves the training state safely.
    """
    torch.save(state, path)

class AverageMeter(object):
    """computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count