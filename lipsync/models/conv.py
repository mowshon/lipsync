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
        residual (bool): If True, adds the input to the output (residual connection). (default is False)

    Attributes:
        conv_block (nn.Sequential): Sequential container with convolution and batch normalization layers.
        act (nn.ReLU): ReLU activation function.
        residual (bool): Whether to apply a residual connection.
    """

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        """
        Forward pass through the Conv2d layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the convolution, batch normalization, and activation.
        """
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class nonorm_Conv2d(nn.Module):
    """
    Convolutional layer without batch normalization, using LeakyReLU as activation.

    Args:
        cin (int): Number of input channels.
        cout (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple): Stride of the convolution.
        padding (int or tuple): Zero-padding added to both sides of the input.
        residual (bool): If True, adds the input to the output (residual connection). (default is False)

    Attributes:
        conv_block (nn.Sequential): Sequential container with convolution.
        act (nn.LeakyReLU): LeakyReLU activation function.
    """

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
        )
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x):
        """
        Forward pass through the nonorm_Conv2d layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the convolution and LeakyReLU activation.
        """
        out = self.conv_block(x)
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
        output_padding (int or tuple): Additional size added to the output shape. (default is 0)

    Attributes:
        conv_block (nn.Sequential): Sequential container with transposed convolution and batch normalization layers.
        act (nn.ReLU): ReLU activation function.
    """

    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        """
        Forward pass through the Conv2dTranspose layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the transposed convolution, batch normalization, and activation.
        """
        out = self.conv_block(x)
        return self.act(out)
