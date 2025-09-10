import torch


def get_activation_fn(name):
    name = name.lower()
    if name == "relu":
        return torch.nn.ReLU()
    elif name == "gelu":
        return torch.nn.GELU()
    elif name == "silu":
        return torch.nn.SiLU()
    elif name == "tanh":
        return torch.tanh
    elif name == 'sigmoid':
        return torch.sigmoid
    elif name == 'softmax':
        return lambda x: torch.softmax(x, dim=-1)
    else:
        raise ValueError(f"Unsupported activation: {name}")
