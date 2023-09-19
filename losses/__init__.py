" Many losses code is from https://github.com/JunMa11/SegLoss/tree/master/losses_pytorch"
from losses.dice import *
from losses.distillation import *

__all__ = ['all_losses', 'get_loss']

all_losses = {
    'CrossEntropyLoss': torch.nn.CrossEntropyLoss,
    'MSE': torch.nn.MSELoss,
    'BCE': torch.nn.BCELoss,
    'ExpLogLoss': ExpLogLoss,
    'DistillationLoss': distillation,
}


def get_loss(name: str, device=None, **kwargs):
    assert name in all_losses, 'loss %s does not exist' % name
    reset_weight_in_config_recursively(kwargs, device)
    return all_losses[name](**kwargs)


def reset_weight_in_config_recursively(config: dict, device):
    for k in config:
        if k == 'weight':
            if torch.is_tensor(config['weight']):
                config[k] = config['weight'].detach().clone().to(device)
            else:
                config[k] = torch.tensor(config['weight'], dtype=torch.float32, device=device)
        elif isinstance(config[k], dict):
            reset_weight_in_config_recursively(config[k], device)
        else:
            continue
