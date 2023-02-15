import torch
from torch import nn

# TODO[#1]: Make Generator and Discriminator
# classes inherit from a base class
# TODO[#2]: Remove normal_init function and use
# the one in basemodel.py


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class ConvBlock(torch.nn.Module):
    r"""
    Convolutional-BatchNorm-ReLU layer with k filters.
    Parameters
    ----------
    in_channels : int
        The number of channels in the input tensor.
    out_channels : int
        The number of channels in the output tensor.
    kernel_size : int, default=4
        The size of the convolutional kernel.
    stride : int, default=2
        The stride of the convolution.
    padding : int, default=1
        The padding of the convolution.
    activation : bool, default=True
        Whether to use a ReLU activation.
    batch_norm : bool, default=True
        Whether to use a batch normalization layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        activation=True,
        batch_norm=True,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.activation = activation
        self.batch_norm = batch_norm
        if self.activation:
            self.relu = nn.LeakyReLU(0.2, inplace=True)
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.activation:
            x = self.relu(x)
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        return x


class DeConvBlock(torch.nn.Module):
    r"""
    Convolutional-BatchNorm-Dropout-ReLU layer with 50% dropout.
    Parameters
    ----------
    in_channels : int
        The number of channels in the input tensor.
    out_channels : int
        The number of channels in the output tensor.
    kernel_size : int, default=4
        The size of the convolutional kernel.
    stride : int, default=2
        The stride of the convolution.
    padding : int, default=1
        The padding of the convolution.
    batch_norm : bool, default=True
        Whether to use a batch normalization layer.
    dropout : bool, default=False
        Whether to use a dropout layer.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        batch_norm=True,
        dropout=False,
    ):
        super(DeConvBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.batch_norm = batch_norm
        self.dropout = dropout
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        if self.dropout:
            self.drop = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.deconv(x)
        if self.batch_norm:
            x = self.bn(x)
        if self.dropout:
            x = self.drop(x)
        return x


class Generator(torch.nn.Module):
    r"""
    Generator network for pix2pix.
    Parameters
    ----------
    nci : int
        The number of channels in the input tensor.
    nco : int, default=None
        The number of channels in the output tensor.
        If None, nco = nci.
    ngf : int, default=64
        The number of filters in the generator.
    """

    def __init__(self, nci, nco=None, ngf=64):
        super(Generator, self).__init__()
        if nco is None:
            nco = nci
        # U-Net encoder
        self.conv1 = ConvBlock(
            in_channels=nci, out_channels=ngf, activation=False, batch_norm=False
        )
        self.conv2 = ConvBlock(in_channels=ngf, out_channels=ngf * 2)
        self.conv3 = ConvBlock(in_channels=ngf * 2, out_channels=ngf * 4)
        self.conv4 = ConvBlock(in_channels=ngf * 4, out_channels=ngf * 8)
        self.conv5 = ConvBlock(in_channels=ngf * 8, out_channels=ngf * 8)
        self.conv6 = ConvBlock(in_channels=ngf * 8, out_channels=ngf * 8)
        self.conv7 = ConvBlock(in_channels=ngf * 8, out_channels=ngf * 8)
        self.conv8 = ConvBlock(
            in_channels=ngf * 8, out_channels=ngf * 8, batch_norm=False
        )

        # U-Net decoder
        self.deconv1 = DeConvBlock(
            in_channels=ngf * 8, out_channels=ngf * 8, dropout=True
        )
        self.deconv2 = DeConvBlock(
            in_channels=ngf * 8 * 2, out_channels=ngf * 8, dropout=True
        )
        self.deconv3 = DeConvBlock(
            in_channels=ngf * 8 * 2, out_channels=ngf * 8, dropout=True
        )
        self.deconv4 = DeConvBlock(in_channels=ngf * 8 * 2, out_channels=ngf * 8)
        self.deconv5 = DeConvBlock(in_channels=ngf * 8 * 2, out_channels=ngf * 4)
        self.deconv6 = DeConvBlock(in_channels=ngf * 4 * 2, out_channels=ngf * 2)
        self.deconv7 = DeConvBlock(in_channels=ngf * 2 * 2, out_channels=ngf)
        self.deconv8 = DeConvBlock(
            in_channels=ngf * 2, out_channels=nco, batch_norm=False
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)
        e7 = self.conv7(e6)
        e8 = self.conv8(e7)

        d1 = self.deconv1(e8)
        d1 = torch.cat((d1, e7), dim=1)
        d2 = self.deconv2(d1)
        d2 = torch.cat((d2, e6), dim=1)
        d3 = self.deconv3(d2)
        d3 = torch.cat((d3, e5), dim=1)
        d4 = self.deconv4(d3)
        d4 = torch.cat((d4, e4), dim=1)
        d5 = self.deconv5(d4)
        d5 = torch.cat((d5, e3), dim=1)
        d6 = self.deconv6(d5)
        d6 = torch.cat((d6, e2), dim=1)
        d7 = self.deconv7(d6)
        d7 = torch.cat((d7, e1), dim=1)
        d8 = self.deconv8(d7)
        return torch.tanh(d8)


class Discriminator(torch.nn.Module):
    r"""
    Discriminator network for pix2pix.
    Parameters
    ----------
    nci : int
        The number of channels in the input tensor.
    nco : int, default=1
        The number of channels in the output tensor.
    ndf : int, default=64
        The number of filters in the discriminator.
    """

    def __init__(self, nci, ndf=64):
        super(Discriminator, self).__init__()
        self.conv1 = ConvBlock(
            in_channels=nci * 2, out_channels=ndf, activation=False, batch_norm=False
        )
        self.conv2 = ConvBlock(in_channels=ndf, out_channels=ndf * 2)
        self.conv3 = ConvBlock(in_channels=ndf * 2, out_channels=ndf * 4)
        self.conv4 = ConvBlock(
            in_channels=ndf * 4, out_channels=ndf * 8, stride=1, padding=1
        )
        self.conv5 = ConvBlock(
            in_channels=ndf * 8, out_channels=1, stride=1, batch_norm=False
        )

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return torch.sigmoid(x)
