import torch

def clip_gradient(optimizer, grad_clip):
    """
        Clips gradients computed during backpropagation to avoid explosion of gradients.
        :param optimizer: optimizer with the gradients to be clipped
        :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * init_lr
        lr = param_group['lr']
    return lr


def create_folder(save_path):
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Create Folder [“{save_path}”].")
    return save_path


def random_seed_setting(seed: int = 42):
    """fixed random seed"""
    import random
    import os
    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def compute_accuracy(predictions, ground_truth):
    predicted_classes = torch.argmax(predictions, dim=1)
    accuracy = torch.sum(predicted_classes == ground_truth).item() / len(ground_truth)
    return accuracy