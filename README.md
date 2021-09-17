# LipSync
Lips Synchronization (Wav2Lip).

## Install
```
git clone git@github.com:mowshon/lipsync.git
cd lipsync
python setup.py install
```

Download the weights
----------
| Model  | Description |  Link to the model | 
| :-------------: | :---------------: | :---------------: |
| Wav2Lip  | Highly accurate lip-sync | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9jtMfbNDaQ?e=TBFBVW)  |
| Wav2Lip + GAN  | Slightly inferior lip-sync, but better visual quality | [Link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW) |

### Project structure
```
└── project-folder
   ├── cache/
   ├── main.py
   ├── wav2lip.pth
   ├── face.mp4
   └── audio.wav
```

## Example
```python
from lipsync import LipSync


lip = LipSync(
    checkpoint_path='wav2lip.pth',  # Downloaded weights
    nosmooth=True,
    cache_dir='cache'  # Cache directory
)

lip.sync(
    'face.mp4',
    'audio.wav',
    'output-file.mp4'
)
```

License and Citation
----------
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


Acknowledgements
----------
Parts of the code structure is inspired by this [TTS repository](https://github.com/r9y9/deepvoice3_pytorch). We thank the author for this wonderful code. The code for Face Detection has been taken from the [face_alignment](https://github.com/1adrianb/face-alignment) repository. We thank the authors for releasing their code and models.