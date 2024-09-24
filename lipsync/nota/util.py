from typing import Dict, Type

import torch

from lipsync.nota import NotaWav2Lip, Wav2Lip, Wav2LipBase

MODEL_REGISTRY: Dict[str, Type[Wav2LipBase]] = {
    'wav2lip': Wav2Lip,
    'nota_wav2lip': NotaWav2Lip
}

def _load(checkpoint_path, device):
    assert device in ['cpu', 'cuda']

    print(f"Load checkpoint from: {checkpoint_path}")
    if device == 'cuda':
        return torch.load(checkpoint_path)
    return torch.load(checkpoint_path, map_location=lambda storage, _: storage)

def load_model(model_name: str, device, checkpoint, **kwargs) -> Wav2LipBase:

    cls = MODEL_REGISTRY[model_name.lower()]
    assert issubclass(cls, Wav2LipBase)

    model = cls(**kwargs)
    checkpoint = _load(checkpoint, device)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    return model.eval()

def count_params(model):
    return sum(p.numel() for p in model.parameters())