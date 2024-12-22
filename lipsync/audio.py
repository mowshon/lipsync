import librosa
import librosa.filters
import numpy as np
from scipy import signal
from lipsync.hparams import HParams

hp = HParams()


def load_wav(path: str, sr: int) -> np.ndarray:
    """Load a WAV file.

    Args:
        path (str): Path to the WAV file.
        sr (int): Sampling rate to load the audio at.

    Returns:
        np.ndarray: Audio time series as a 1D numpy array.
    """
    return librosa.core.load(path, sr=sr)[0]


def preemphasis_func(wav: np.ndarray, k: float, preemphasize: bool = True) -> np.ndarray:
    """Apply a preemphasis filter to the waveform.

    Args:
        wav (np.ndarray): Input waveform as a 1D numpy array.
        k (float): Preemphasis coefficient.
        preemphasize (bool): Whether to apply preemphasis or not.

    Returns:
        np.ndarray: Preemphasized or original waveform.
    """
    # The filter: y[n] = x[n] - k*x[n-1]
    # This increases the magnitude of high-frequency components.
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def melspectrogram(wav: np.ndarray) -> np.ndarray:
    """Compute the mel-spectrogram of a waveform.

    Args:
        wav (np.ndarray): Input waveform array.

    Returns:
        np.ndarray: Mel-spectrogram as a 2D numpy array (num_mels x time).
    """
    D = _stft(preemphasis_func(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(_linear_to_mel(np.abs(D))) - hp.ref_level_db

    if hp.signal_normalization:
        return _normalize(S)
    return S


def _stft(y: np.ndarray) -> np.ndarray:
    """Compute the STFT of the given waveform.

    Args:
        y (np.ndarray): Input waveform.

    Returns:
        np.ndarray: Complex STFT of y. Shape is (1 + n_fft/2, time).
    """
    return librosa.stft(y=y, n_fft=hp.n_fft, hop_length=hp.hop_size, win_length=hp.win_size)


def _linear_to_mel(spectrogram: np.ndarray) -> np.ndarray:
    """Convert a linear-scale spectrogram to mel-scale.

    Args:
        spectrogram (np.ndarray): Linear frequency spectrogram.

    Returns:
        np.ndarray: Mel-scale spectrogram.
    """
    mel_basis = _build_mel_basis()
    return np.dot(mel_basis, spectrogram)


def _build_mel_basis() -> np.ndarray:
    """Construct a mel-filter bank.

    Returns:
        np.ndarray: Mel filter bank matrix.
    """
    assert hp.fmax <= hp.sample_rate // 2
    return librosa.filters.mel(
        sr=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.fmin,
        fmax=hp.fmax
    )


def _amp_to_db(x: np.ndarray) -> np.ndarray:
    """Convert amplitude to decibels.

    Args:
        x (np.ndarray): Amplitude values.

    Returns:
        np.ndarray: Decibel-scaled values.
    """
    min_level = np.exp(hp.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(spec: np.ndarray) -> np.ndarray:
    """Normalize the mel-spectrogram.

    Args:
        spec (np.ndarray): Decibel-scaled mel-spectrogram.

    Returns:
        np.ndarray: Normalized mel-spectrogram.
    """
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            # Rescale to symmetric range [-max_abs_value, max_abs_value]
            return np.clip(
                (2 * hp.max_abs_value) * ((spec - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                -hp.max_abs_value,
                hp.max_abs_value
            )
        else:
            # Rescale to [0, max_abs_value]
            return np.clip(
                hp.max_abs_value * ((spec - hp.min_level_db) / (-hp.min_level_db)),
                0,
                hp.max_abs_value
            )

    # If no clipping is allowed, validate ranges
    assert spec.max() <= 0 and spec.min() - hp.min_level_db >= 0

    if hp.symmetric_mels:
        # Symmetric range normalization
        return (2 * hp.max_abs_value) * ((spec - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    else:
        # Asymmetric range normalization
        return hp.max_abs_value * ((spec - hp.min_level_db) / (-hp.min_level_db))
