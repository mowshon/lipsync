"""
Model registration and loading utilities for lipsync.
"""

from typing import Dict, Any
import torch

from lipsync.models.wav2lip import Wav2Lip

# A registry of available models
MODEL_REGISTRY: Dict[str, Any] = {
    'wav2lip': Wav2Lip,
}


def _load_checkpoint(checkpoint_path: str, device: str) -> Any:
    """
    Loads a model checkpoint from the specified path onto the given device.

    Args:
        checkpoint_path (str): The path to the model checkpoint file.
        device (str): The device to load the model on, either 'cpu' or 'cuda'.

    Returns:
        Any: The loaded model checkpoint.

    Raises:
        AssertionError: If the device is not 'cpu' or 'cuda'.
        FileNotFoundError: If checkpoint file not found.
    """
    # Ensure valid device
    assert device in ['cpu', 'cuda'], "Device must be 'cpu' or 'cuda'"

    # Try loading checkpoint on the specified device
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device(device)
        )
        return checkpoint
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.") from e


def load_model(model_name: str, device: str, checkpoint: str) -> torch.nn.Module:
    """
    Loads and initializes a model with the given checkpoint and device.

    Args:
        model_name (str): The name of the model to load, e.g. 'wav2lip'.
        device (str): The device to load the model on, either 'cpu' or 'cuda'.
        checkpoint (str): The path to the model checkpoint.

    Returns:
        torch.nn.Module: The initialized and evaluated model.

    Raises:
        KeyError: If the model name is not found in the model registry.
    """
    # Retrieve the model class from the registry, converting name to lower in case of mismatch
    cls = MODEL_REGISTRY[model_name.lower()]

    # Initialize the model
    model = cls()

    # Load the checkpoint
    checkpoint_dict = _load_checkpoint(checkpoint, device)

    # Load state dict into model
    model.load_state_dict(checkpoint_dict)

    # Move model to the specified device
    model = model.to(device)

    # Put model into evaluation mode
    return model.eval()


__all__ = [
    'Wav2Lip',
    'load_model'
]
