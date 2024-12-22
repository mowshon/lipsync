class HParams:
    """
    Hyperparameters for audio and video processing.

    Attributes:
        num_mels (int): Number of mel-spectrogram channels and local conditioning dimensionality.
        n_fft (int): Extra window size is filled with 0 paddings to match this parameter.
        hop_size (int): Hop size in samples. For 16000Hz, 200 samples correspond to 12.5 ms.
        win_size (int): Window size in samples. For 16000Hz, 800 samples correspond to 50 ms.
        sample_rate (int): Sampling rate in Hz, default is 16000Hz, corresponding to LibriSpeech.
        signal_normalization (bool): Whether to normalize the mel spectrograms.
        allow_clipping_in_normalization (bool): Allow clipping during normalization. Relevant only if signal_normalization is True.
        symmetric_mels (bool): Scale data symmetrically around 0 for faster convergence.
        max_abs_value (float): Maximum absolute value for normalization.
        preemphasize (bool): Whether to apply a pre-emphasis filter to reduce spectrogram noise.
        preemphasis (float): Pre-emphasis filter coefficient.
        min_level_db (int): Minimum level in decibels for the signal.
        ref_level_db (int): Reference level in decibels for the signal.
        fmin (int): Minimum frequency in Hz. Set to 55 for male speakers and higher for female speakers.
        fmax (int): Maximum frequency in Hz. Adjust based on the dataset.
        img_size (int): Size of the input images for video processing.
        fps (int): Frames per second for the video.
    """

    num_mels = 80

    n_fft = 800
    hop_size = 200
    win_size = 800
    sample_rate = 16000

    signal_normalization = True
    allow_clipping_in_normalization = True
    symmetric_mels = True
    max_abs_value = 4.0
    preemphasize = True
    preemphasis = 0.97

    min_level_db = -100
    ref_level_db = 20
    fmin = 55
    fmax = 7600

    img_size = 96
    fps = 25
