import math

from torch import nn

import ganetic.basemodel as basemodel


class Generator(basemodel.Generator):
    r"""Deep Convolutional Generative Adversarial Network (DCGAN) Generator.
    Parameters
    ----------
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
        nz=100,
        nc=3,
        ngf=64,
        out_size=64,
        activation=nn.ReLU(True),
        last_activation=nn.Tanh()
    ):
        super(Generator, self).__init__()
        if out_size < 16 and math.ceil(math.log2(out_size)) != math.floor(math.log2(out_size)):
            raise Exception(
                "out_size must be a power of 2 and greater than 16")
        total_repeats = out_size.bit_length() - 4
        model = []
        _ngf = ngf * 2 ** total_repeats
        model += [
            nn.Sequential(
                nn.ConvTranspose2d(nz, _ngf, 4, 1, 0, bias=False),
                nn.BatchNorm2d(_ngf),
                activation
            )
        ]
        for _ in range(total_repeats):
            _ngf //= 2
            model += [
                nn.Sequential(
                    nn.ConvTranspose2d(_ngf * 2, _ngf, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(_ngf),
                    activation
                )
            ]
        model += [
            nn.Sequential(
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                last_activation
            )
        ]
        self.model = nn.Sequential(*model)
        self._weights_init(self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), x.size(1), 1, 1))


class Discriminator(basemodel.Discriminator):
    r"""Deep Convolutional Generative Adversarial Network (DCGAN) Discriminator.
    Parameters
    ----------
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
        nc=3,
        ndf=64,
        in_size=64,
        activation=nn.LeakyReLU(0.2, inplace=True),
        last_activation=nn.Sigmoid()
    ):
        super(Discriminator, self).__init__()
        if in_size < 16 and math.ceil(math.log2(in_size)) != math.floor(math.log2(in_size)):
            raise Exception(
                "in_size must be a power of 2 and greater than 16")
        total_repeats = in_size.bit_length() - 4
        model = []
        _ndf = ndf
        model += [
            nn.Sequential(
                nn.Conv2d(nc, _ndf, 4, 2, 1, bias=False),
                activation
            )
        ]
        for _ in range(total_repeats):
            model += [
                nn.Sequential(
                    nn.Conv2d(_ndf, _ndf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(_ndf * 2),
                    activation
                )
            ]
            _ndf *= 2
        model += [
            nn.Sequential(
                nn.Conv2d(_ndf, 1, 4, 1, 0, bias=False),
                last_activation
            )
        ]
        self.model = nn.Sequential(*model)
        self._weights_init(self.model)

    def forward(self, x):
        return self.model(x).view(-1, 1).squeeze(1)
