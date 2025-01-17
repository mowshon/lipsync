"""
Definition of the Wav2Lip model, which generates lip-synced video frames from audio
and face frames. It includes both the audio encoder and the face encoder/decoder modules.
"""

import torch
from torch import nn

from lipsync.models.conv import Conv2dTranspose, Conv2d


class Wav2Lip(nn.Module):
    """
    Wav2Lip model for generating lip-synced videos.

    This model takes as input sequences of audio and corresponding face frames and produces
    synthesized video frames where the lip movements are synchronized with the given audio.
    """

    def __init__(self):
        """
        Initializes the Wav2Lip model modules:
        - Face encoder blocks
        - Audio encoder
        - Face decoder blocks
        - Output block
        """
        super(Wav2Lip, self).__init__()

        # Face encoder blocks
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(
                Conv2d(6, 16, kernel_size=7, stride=1, padding=3)
            ),
            nn.Sequential(
                Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)
            ),
            nn.Sequential(
                Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)
            ),
            nn.Sequential(
                Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)
            ),
            nn.Sequential(
                Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)
            ),
            nn.Sequential(
                Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)
            ),
            nn.Sequential(
                Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
                Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
            ),
        ])

        # Audio encoder
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
        )

        # Face decoder blocks
        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(
                Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
            ),
            nn.Sequential(
                Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)
            ),
            nn.Sequential(
                Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)
            ),
            nn.Sequential(
                Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True)
            ),
            nn.Sequential(
                Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)
            ),
            nn.Sequential(
                Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)
            ),
            nn.Sequential(
                Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)
            ),
        ])

        # Output block
        self.output_block = nn.Sequential(
            Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(
        self,
        audio_sequences: torch.Tensor,
        face_sequences: torch.Tensor
    ) -> torch.Tensor:
        """
        Runs the forward pass of the Wav2Lip model.

        Args:
            audio_sequences (torch.Tensor): The input audio sequences of shape (B, T, 1, 80, 16)
                or (B, 1, 80, 16) if no time dimension.
            face_sequences (torch.Tensor): The corresponding face image sequences of shape
                (B, C, T, H, W) or (B, C, H, W) if no time dimension.

        Returns:
            torch.Tensor: The output video frames with lips synchronized to the audio. The shape
                will be (B, C, T, H, W) if input has time dimension, otherwise (B, C, H, W).
        """
        B = audio_sequences.size(0)
        input_dim_size = face_sequences.dim()

        # Reshape sequences if input is batched over time
        if input_dim_size > 4:
            BT = B * audio_sequences.size(1)
            audio_sequences = audio_sequences.view(
                BT,
                audio_sequences.size(2),
                audio_sequences.size(3),
                audio_sequences.size(4)
            )

            # Face shape: (B, C, T, H, W) => (B, T, C, H, W) => (BT, C, H, W)
            _, C, T, H, W = face_sequences.size()
            face_sequences = face_sequences.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, H, W)

        # Encode audio
        audio_embedding = self.audio_encoder(audio_sequences)

        # Encode face and store intermediate features
        feats = []
        x = face_sequences
        for block in self.face_encoder_blocks:
            x = block(x)
            feats.append(x)
        feats.reverse()  # Reverse the order for decoder

        # Decode with audio embedding
        x = audio_embedding
        for i, block in enumerate(self.face_decoder_blocks):
            x = block(x)
            x = torch.cat((x, feats[i]), dim=1)

        # Generate output
        x = self.output_block(x)

        # Reshape back to original format if needed
        if input_dim_size > 4:
            total_frames = audio_embedding.size(0)
            T = total_frames // B
            x = x.view(B, T, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        return x
