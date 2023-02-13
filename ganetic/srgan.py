import math

import torch
from torch import nn

import ganetic.basemodel as basemodel

from .layers import ResidualBlock, UpsampleBlock

# TODO[#1]: Make Generator and Discriminator
# classes inherit from a base class


class Generator(basemodel.Generator):
    r"""
    Parameters
    ----------
    scale_factor : int
        The scale factor for the super resolution.
    nci : int
        The number of channels in the input tensor.
    nco : int
        The number of channels in the output tensor.
    ngf : int, default=64
        The number of filters in the generator.
    no_of_residual_blocks : int, default=5
        The number of residual blocks in the generator.
    """

    def __init__(
        self,
        scale_factor,
        nci=3,
        nco=3,
        ngf=64,
        no_of_residual_blocks=5
    ):
        super(Generator, self).__init__()
        upsample_block_num = int(math.log(scale_factor, 2))

        self.block1 = nn.Sequential(
            nn.Conv2d(nci, ngf, kernel_size=9, padding=4),
            nn.PReLU()
        )
        block2 = [ResidualBlock(ngf)
                  for _ in range(no_of_residual_blocks)]
        self.block2 = nn.Sequential(*block2)
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf, ngf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ngf)
        )
        block4 = [UpsampleBlock(ngf, 2)
                  for _ in range(upsample_block_num)]
        block4.append(nn.Conv2d(ngf, nco, kernel_size=9, padding=4))
        self.block4 = nn.Sequential(*block4)

    def forward(self, x):
        x = self.block1(x)
        block2 = x
        block2 = self.block2(block2)
        block3 = self.block3(block2) + x
        block4 = self.block4(block3)
        return (torch.tanh(block4) + 1) / 2


class Discriminator(basemodel.Discriminator):
    r"""
    Parameters
    ----------
    input_shape : tuple
        The shape of the input tensor.
    ndf : int, default=64
        The number of filters in the discriminator.
    negative_slope : float, default=0.2
        The negative slope of the leaky relu.
    """

    def __init__(
        self,
        input_shape,
        ndf=64,
        negative_slope=0.2,
    ):
        super(Discriminator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_shape[0], ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(ndf, ndf, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
        self.block8 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
        with torch.no_grad():
            linear_size = self.block8(self.block7(
                self.block6(self.block5(
                    self.block4(self.block3(
                        self.block2(self.block1(
                            torch.zeros(1, *(input_shape)))))))))).view(1, -1).size(1)
        self.block9 = nn.Sequential(
            nn.Linear(linear_size, 1024),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        )
        self.block10 = nn.Sequential(
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = x.view(x.size(0), -1)
        x = self.block9(x)
        x = self.block10(x)
        return x
