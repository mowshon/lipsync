# lipsync

lipsync is a Python library that moves lips in a video (or image) to match a given audio file. It is based on [Wav2Lip](https://github.com/Rudrabha/Wav2Lip), but many unneeded files and libraries have been removed, and the code has been updated to work with the latest versions of Python.

---

## Features

- **Video lip synchronization**  
  Synchronize lips in an existing video to match a new audio file.

- **Image lip animation**  
  Provide a single image and an audio file to generate a talking video.

- **Runs on CPU and CUDA**  
  You can choose whether to run on your CPU or a CUDA-enabled GPU for faster processing.

- **Caching**  
  If you use the same video multiple times with different audio files, lipsync can cache frames and reuse them. This makes future runs much faster.

---

## Pre-Trained Weights

lipsync works with two different pre-trained models:

1. **Wav2Lip ([Download wav2lip.pth](https://drive.google.com/file/d/1qKU8HG8dR4nW4LvCqpEYmSy6LLpVkZ21/view?usp=sharing))**  
   - More accurate lip synchronization  
   - Lips in the result may appear somewhat blurred

2. **Wav2Lip + GAN ([Download wav2lip_gan.pth](https://drive.google.com/file/d/13Ktexq-nZOsAxqrTdMh3Q0ntPB3yiBtv/view?usp=sharing))**  
   - Lips in the result are clearer  
   - Synchronization may be slightly less accurate

---

## Installation

```bash
pip install lipsync
```

## Usage Example

Below is a simple example in Python. This assumes you have the model weights (either `wav2lip.pth` or `wav2lip_gan.pth`) in a `weights/` folder.

```python
from lipsync import LipSync

lip = LipSync(
    model='wav2lip',
    checkpoint_path='weights/wav2lip.pth',
    nosmooth=True,
    device='cuda',
    cache_dir='cache',
    img_size=96,
    save_cache=True,
)

lip.sync(
    'source/person.mp4',
    'source/audio.wav',
    'result.mp4',
)
```

### Important Parameters
- model: `'wav2lip'` or `'wav2lip_gan'`
- checkpoint_path: Path to the model weights (e.g., `wav2lip.pth`, `wav2lip_gan.pth`)
- nosmooth: Set `True` to disable smoothing
- device: `'cpu'` or `'cuda'`
- cache_dir: Directory for saving frames
- save_cache: Set `True` to save frames to `cache_dir` for faster re-runs


### Ethical Use

Please be mindful when using **lipsync**. This library can generate videos that look convincing, so it could be used to spread disinformation or harm someone’s reputation. We encourage using it only for **entertainment** or **scientific** purposes, and always with **respect and consent** from any people involved.

### License and Citation

The software can only be used for personal/research/non-commercial purposes. Please cite the following paper if you have use this code:
```
@inproceedings{10.1145/3394171.3413532,
    author = {Prajwal, K R and Mukhopadhyay, Rudrabha and Namboodiri, Vinay P. and Jawahar, C.V.},
    title = {A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild},
    year = {2020},
    isbn = {9781450379885},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3394171.3413532},
    doi = {10.1145/3394171.3413532},
    booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
    pages = {484–492},
    numpages = {9},
    keywords = {lip sync, talking face generation, video generation},
    location = {Seattle, WA, USA},
    series = {MM '20}
}
```
