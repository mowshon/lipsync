import torch
from torch import nn

from lipsync.nota.base import Wav2LipBase
from lipsync.nota.conv import Conv2d, Conv2dTranspose


class NotaWav2Lip(Wav2LipBase):
    def __init__(self, nef=4, naf=8, ndf=8, x_size=96):
        super().__init__()

        assert x_size in [96, 128]
        self.ker_sz_last = x_size // 32

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, nef, kernel_size=7, stride=1, padding=3)),  # 96,96

            nn.Sequential(Conv2d(nef, nef * 2, kernel_size=3, stride=2, padding=1),),  # 48,48

            nn.Sequential(Conv2d(nef * 2, nef * 4, kernel_size=3, stride=2, padding=1),),  # 24,24

            nn.Sequential(Conv2d(nef * 4, nef * 8, kernel_size=3, stride=2, padding=1),),  # 12,12

            nn.Sequential(Conv2d(nef * 8, nef * 16, kernel_size=3, stride=2, padding=1),),  # 6,6

            nn.Sequential(Conv2d(nef * 16, nef * 32, kernel_size=3, stride=2, padding=1),),  # 3,3

            nn.Sequential(Conv2d(nef * 32, nef * 32, kernel_size=self.ker_sz_last, stride=1, padding=0),  # 1, 1
                          Conv2d(nef * 32, nef * 32, kernel_size=1, stride=1, padding=0)), ])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, naf, kernel_size=3, stride=1, padding=1),

            Conv2d(naf, naf * 2, kernel_size=3, stride=(3, 1), padding=1),

            Conv2d(naf * 2, naf * 4, kernel_size=3, stride=3, padding=1),

            Conv2d(naf * 4, naf * 8, kernel_size=3, stride=(3, 2), padding=1),

            Conv2d(naf * 8, naf * 16, kernel_size=3, stride=1, padding=0),
            Conv2d(naf * 16, naf * 16, kernel_size=1, stride=1, padding=0), )

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(naf * 16, naf * 16, kernel_size=1, stride=1, padding=0), ),

            nn.Sequential(Conv2dTranspose(nef * 32 + naf * 16, ndf * 16, kernel_size=self.ker_sz_last, stride=1, padding=0),),
                          # 3,3 # 512+512 = 1024

            nn.Sequential(
                Conv2dTranspose(nef * 32 + ndf * 16, ndf * 16, kernel_size=3, stride=2, padding=1, output_padding=1),),  # 6, 6
                # 512+512 = 1024

            nn.Sequential(
                Conv2dTranspose(nef * 16 + ndf * 16, ndf * 12, kernel_size=3, stride=2, padding=1, output_padding=1),),  # 12, 12
                # 256+512 = 768

            nn.Sequential(
                Conv2dTranspose(nef * 8 + ndf * 12, ndf * 8, kernel_size=3, stride=2, padding=1, output_padding=1),),  # 24, 24
                # 128+384 = 512

            nn.Sequential(
                Conv2dTranspose(nef * 4 + ndf * 8, ndf * 4, kernel_size=3, stride=2, padding=1, output_padding=1),),  # 48, 48
                # 64+256 = 320

            nn.Sequential(
                Conv2dTranspose(nef * 2 + ndf * 4, ndf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),), # 96,96
                # 32+128 = 160
        ])

        self.output_block = nn.Sequential(Conv2d(nef + ndf * 2, ndf, kernel_size=3, stride=1, padding=1),  # 16+64 = 80
                                          nn.Conv2d(ndf, 3, kernel_size=1, stride=1, padding=0),
                                          nn.Sigmoid())