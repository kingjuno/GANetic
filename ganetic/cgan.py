from torch import nn
from .dcgan import Generator as DCGANGenerator, Discriminator as DCGANDiscriminator
import torch


class Generator(DCGANGenerator):
    r"""Conditional Generative Adversarial Network (CGAN) Generator.
    Parameters
    ----------
    n_classes : int
        Number of classes.
    nz : int, default 100
        Size of the latent z vector.
    nc : int, default 3
        Number of channels in the output image.
    ngf: int, default 64
        Size of feature maps in generator.
    out_size: int, default 64
        Size of the output image.
    activation: torch.nn.Module, default nn.ReLU(True)
        Activation function to use in the generator.
    last_activation: torch.nn.Module, default nn.Tanh()
        Activation function to use in the last layer of the generator.
    """

    def __init__(
        self,
        n_classes,
        nz=100,
        nc=3,
        ngf=64,
        out_size=64,
        activation=nn.ReLU(True),
        last_activation=nn.Tanh(),
    ):
        super(Generator, self).__init__(
            nz + n_classes, nc, ngf, out_size, activation, last_activation
        )
        self.n_classes = n_classes
        self.nz = nz
        self.label_embedding = nn.Embedding(n_classes, n_classes)

    def forward(self, x, labels):
        label_output = self.label_embedding(labels)
        return super(Generator, self).forward(torch.cat((x, label_output), 1))


class Discriminator(DCGANDiscriminator):
    r"""
    Parameters
    ----------
    n_classes : int
        Number of classes.
    nc : int, default 3
        Number of channels in the input image.
    ndf: int, default 64
        Size of feature maps in discriminator.
    in_size: int, default 64
        Size of the input image.
    activation: torch.nn.Module, default nn.LeakyReLU(0.2, inplace=True)
        Activation function to use in the discriminator.
    last_activation: torch.nn.Module, default nn.Sigmoid()
        Activation function to use in the last layer of the discriminator.
    """

    def __init__(
        self,
        n_classes,
        nc=3,
        ndf=64,
        in_size=64,
        activation=nn.LeakyReLU(0.2, inplace=True),
        last_activation=nn.Sigmoid(),
    ):
        super(Discriminator, self).__init__(
            nc + n_classes, ndf, in_size, activation, last_activation
        )
        self.n_classes = n_classes
        self.label_embedding = nn.Embedding(n_classes, n_classes)

    def forward(self, x, labels):
        label_output = self.label_embedding(labels)
        label = (
            label_output.unsqueeze(2)
            .unsqueeze(3)
            .expand(label_output.size(0), label_output.size(1), x.size(2), x.size(3))
        )
        return super(Discriminator, self).forward(torch.cat((x, label), 1))
