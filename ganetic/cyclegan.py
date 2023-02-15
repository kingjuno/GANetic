from torch import nn
from torch.nn import functional as F

from ganetic import basemodel


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        _x = x
        x = self.block(x)
        return x + _x


class Generator(basemodel.Generator):
    """
    CycleGAN Generator
    Paper: https://arxiv.org/abs/1703.10593
    Architecture:
        Let C7s1-k denote a 7×7 Convolution-InstanceNorm-ReLU
        layer with k filters and stride 1, dk denotes a 3×3
        Convolution-InstanceNorm-ReLU layer with k filter and
        stride 2, Rk denotes a residual block that contains two
        3×3 convolutional layers with the same number of filters
        on both layer, uk denotes a 3×3 fractional-strided-Covolution-
        InstanceNorm-ReLU layer with k filters and stride 1/2,

        C7s1-64 => d128 => d256 => R256 => R256
        => R256 => R256 => R256 => R256 => R256
        => R256 => R256 => u128 => u64 => C7s1-3

        (with 9 residual blocks)
    Parameters
    ----------
    nci : int, default 3
        Number of channels in the input image.
    nco : int, default 3
        Number of channels in the output image.
    ngf : int, default 64
        Size of the feature maps in the generator.
    no_of_residual_blocks : int, default 9
        Number of residual blocks in the generator.
    activation : torch.nn.Module, default nn.ReLU(True)
        Activation function to use in the generator.
    last_activation : torch.nn.Module, default nn.Tanh()
        Activation function to use in the last layer of the generator.
    """

    def __init__(
        self,
        nci=3,
        nco=3,
        ngf=64,
        no_of_residual_blocks=9,
        activation=nn.ReLU(True),
        last_activation=nn.Tanh(),
    ):
        super(Generator, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(nci, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            activation,
        ]

        # Downsampling
        for i in range(2):
            model += [
                nn.Conv2d(
                    ngf * 2**i, ngf * 2 ** (i + 1), kernel_size=3, stride=2, padding=1
                ),
                nn.InstanceNorm2d(ngf * 2 ** (i + 1)),
                activation,
            ]

        # Residual blocks
        for _ in range(no_of_residual_blocks):
            model += [ResidualBlock(ngf * 4)]

        # Upsampling
        for i in range(2):
            model += [
                nn.ConvTranspose2d(
                    ngf * 2 ** (2 - i),
                    ngf * 2 ** (1 - i),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.InstanceNorm2d(ngf * 2 ** (1 - i)),
                activation,
            ]

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, nco, kernel_size=7, padding=0),
            last_activation,
        ]

        self.model = nn.Sequential(*model)
        self._weights_init(self.model)

    def forward(self, x):
        return self.model(x)


class Discriminator(basemodel.Discriminator):
    """
    CycleGAN Discriminator
    Paper: https://arxiv.org/abs/1703.10593
    Architecture:
        C64-C128-C256-C512

        (with 70x70 PatchGAN)
    Parameters
    ----------
    nci : int, default 3
        Number of channels in the input image.
    ndf : int, default 64
        Size of the feature maps in the discriminator.
    activation : torch.nn.Module, default nn.LeakyReLU(0.2, True)
        Activation function to use in the discriminator.
    last_activation : torch.nn.Module, default nn.Sigmoid()
        Activation function to use in the last layer of the discriminator.
    """

    def __init__(
        self,
        nci=3,
        ndf=64,
        activation=nn.LeakyReLU(0.2, True),
        last_activation=nn.Sigmoid(),
    ):
        super(Discriminator, self).__init__()
        model = [
            nn.Conv2d(nci, ndf, kernel_size=4, stride=2, padding=1),
            activation,
        ]

        for i in range(1, 3):
            model += [
                nn.Conv2d(
                    ndf * 2 ** (i - 1), ndf * 2**i, kernel_size=4, stride=2, padding=1
                ),
                nn.InstanceNorm2d(ndf * 2**i),
                activation,
            ]

        model += [
            nn.Conv2d(ndf * 2**2, ndf * 2**3, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(ndf * 2**3),
            activation,
            nn.Conv2d(ndf * 2**3, 1, kernel_size=4, stride=1, padding=1),
            last_activation,
        ]

        self.model = nn.Sequential(*model)
        self._weights_init(self.model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(-1)
