from lipsync.models.wav2lip import Wav2Lip
from typing import Dict, Any
import torch


# Registry for available models
MODEL_REGISTRY: Dict[str, Any] = {
    'wav2lip': Wav2Lip,
}


def _load(checkpoint_path, device):
    """
    Loads a model checkpoint from the specified path.

    Args:
        checkpoint_path (str): The path to the model checkpoint file.
        device (str): The device to load the model on, either 'cpu' or 'cuda'.

    Returns:
        Any: The loaded model checkpoint.

    Raises:
        AssertionError: If the device is not 'cpu' or 'cuda'.
    """
    assert device in ['cpu', 'cuda'], "Device must be 'cpu' or 'cuda'"

    # Load checkpoint on the specified device
    if device == 'cuda':
        return torch.load(checkpoint_path, weights_only=True)

    return torch.load(
        checkpoint_path,
        map_location=lambda storage, _: storage,
        weights_only=True
    )


def load_model(model_name: str, device: str, checkpoint: str):
    """
    Loads and initializes a model with the given checkpoint and device.

    Args:
        model_name (str): The name of the model to load.
        device (str): The device to load the model on, either 'cpu' or 'cuda'.
        checkpoint (str): The path to the model checkpoint.

    Returns:
        torch.nn.Module: The initialized and evaluated model.

    Raises:
        KeyError: If the model name is not found in the model registry.
    """
    # Retrieve the model class from the registry
    cls = MODEL_REGISTRY[model_name.lower()]

    # Initialize the model
    model = cls()

    # Load the checkpoint and set model state
    checkpoint = _load(checkpoint, device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    return model.eval()


__all__ = [
    'Wav2Lip',
    'load_model'
]
