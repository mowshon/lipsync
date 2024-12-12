import torch
from torch import nn


class Conv2d(nn.Module):
    """
    Convolutional layer with batch normalization and optional residual connection.

    Args:
        cin (int): Number of input channels.
        cout (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple): Stride of the convolution.
        padding (int or tuple): Zero-padding added to both sides of the input.
        residual (bool): If True, adds the input to the output (residual connection). Default is False.

    Attributes:
        conv_block (nn.Sequential): Convolution and batch normalization layers.
        act (nn.ReLU): ReLU activation function.
        residual (bool): Whether to apply a residual connection.
    """

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x)
        if self.residual:
            out = out + x
        return self.act(out)


class Conv2dTranspose(nn.Module):
    """
    Transposed convolutional layer (also known as deconvolution) with batch normalization.

    Args:
        cin (int): Number of input channels.
        cout (int): Number of output channels.
        kernel_size (int or tuple): Size of the transposed convolutional kernel.
        stride (int or tuple): Stride of the transposed convolution.
        padding (int or tuple): Zero-padding added to both sides of the input.
        output_padding (int or tuple): Additional size added to the output shape. Default is 0.

    Attributes:
        conv_block (nn.Sequential): Transposed convolution and batch normalization layers.
        act (nn.ReLU): ReLU activation function.
    """

    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x)
        return self.act(out)
