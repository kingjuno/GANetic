from torch import nn


class Generator(nn.Module):
    r"""
    Base class for all Generators.
    Parameters
    ----------
    nz : int, default 100
        Size of the latent z vector.
    nc : int, default 3
        Number of channels in the training images.
    """

    def __init__(self, nz=100, nc=3):
        super(Generator, self).__init__()
        self.nz = nz
        self.nc = nc

    def _weights_init(self, m):
        r"""
        Default weights initialization for all Generators.
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    r"""
    Base class for all Discriminators.
    Parameters
    ----------
    nc : int, default 3
        Number of channels in the training images.
    """

    def __init__(self, nc=3):
        super(Discriminator, self).__init__()
        self.nc = nc

    def _weights_init(self, m):
        r"""
        Default weights initialization for all Discriminators.
        """
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
