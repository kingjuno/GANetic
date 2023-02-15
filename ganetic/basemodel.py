from torch import nn


class Generator(nn.Module):
    r"""
    Base class for all Generators.
    """

    def __init__(self):
        super(Generator, self).__init__()

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
    """

    def __init__(self):
        super(Discriminator, self).__init__()

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
