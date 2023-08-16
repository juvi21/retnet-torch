from torch.nn import ReLU, GELU, SiLU

ACTIVATION_LAYERS = {
    'relu': ReLU,
    'gelu': GELU,
    'silu': SiLU
}

DEFAULT_ACTIVATION = 'gelu'

def get_activation_fn(activation: str = DEFAULT_ACTIVATION):
    """
    Returns the activation function module based on the provided string.
    """
    if isinstance(activation, str):
        return ACTIVATION_LAYERS.get(activation, GELU)()
    return activation