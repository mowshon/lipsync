import torch
from torch import nn
from lipsync.models.conv import Conv2dTranspose, Conv2d


class Wav2Lip(nn.Module):
    """Wav2Lip model for generating lip-synced videos.

    The Wav2Lip model takes audio sequences and corresponding face sequences to produce
    synchronized video frames where the lips match the input audio.

    Attributes:
        face_encoder_blocks (nn.ModuleList): Encoder blocks for processing face sequences.
        audio_encoder (nn.Sequential): Encoder network for processing audio sequences.
        face_decoder_blocks (nn.ModuleList): Decoder blocks for generating face outputs.
        output_block (nn.Sequential): Final layers producing the output video frames.
    """

    def __init__(self):
        """Initializes the Wav2Lip model."""
        super(Wav2Lip, self).__init__()

        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(
                Conv2d(6, 16, kernel_size=7, stride=1, padding=3)
            ),  # Output size: 96x96

            nn.Sequential(
                Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output size: 48x48
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output size: 24x24
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output size: 12x12
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output size: 6x6
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # Output size: 3x3
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                Conv2d(512, 512, kernel_size=3, stride=1, padding=0),  # Output size: 1x1
                Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
            ),
        ])

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

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(
                Conv2d(512, 512, kernel_size=1, stride=1, padding=0)
            ),

            nn.Sequential(
                Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),  # Output size: 3x3
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)
            ),

            nn.Sequential(
                Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)
            ),  # Output size: 6x6

            nn.Sequential(
                Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True)
            ),  # Output size: 12x12

            nn.Sequential(
                Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)
            ),  # Output size: 24x24

            nn.Sequential(
                Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)
            ),  # Output size: 48x48

            nn.Sequential(
                Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)
            ),  # Output size: 96x96
        ])

        self.output_block = nn.Sequential(
            Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, audio_sequences, face_sequences):
        """Performs a forward pass of the Wav2Lip model.

        Args:
            audio_sequences (torch.Tensor): Audio sequences with shape (B, T, 1, 80, 16).
            face_sequences (torch.Tensor): Face sequences corresponding to the audio.

        Returns:
            torch.Tensor: Generated video frames with lip movements synchronized to the audio.
        """
        B = audio_sequences.size(0)
        input_dim_size = len(face_sequences.size())

        if input_dim_size > 4:
            # Reshape audio and face sequences for processing
            audio_sequences = torch.cat(
                [audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat(
                [face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        # Encode the audio sequences
        audio_embedding = self.audio_encoder(audio_sequences)  # Shape: (B, 512, 1, 1)

        # Encode the face sequences and store intermediate features
        feats = []
        x = face_sequences
        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        # Decode and combine features from audio and face encoders
        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(f"Error in concatenation: {e}")
                print(f"x size: {x.size()}, feats[-1] size: {feats[-1].size()}")
                raise e
            feats.pop()

        # Generate the output frames
        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)  # List of tensors with shape (B, C, H, W)
            outputs = torch.stack(x, dim=2)  # Shape: (B, C, T, H, W)
        else:
            outputs = x

        return outputs
