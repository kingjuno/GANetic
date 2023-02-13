# GANetic

A collection of highly customizable GANs implemented in PyTorch.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [DCGAN](#dcgan)
  - [SRGAN](#srgan)
  - [Pix2Pix](#pix2pix)
  - [Conditional GANs](#conditional-gans)
- [Citations](#citations)

## Installation

Stable version:
```bash
pip install ganetic
```

Latest version:
```bash
pip install git+https://github.com/kingjuno/ganetic.git
```

## Usage

### DCGAN

```python
import torch

from ganetic.dcgan import Discriminator, Generator

netG = Generator(
    nz=100,  # length of latent vector
    nc=3,    # number of channels in the training images.
    ngf=64,  # size of feature maps in generator
)
netD = Discriminator(
    nc=3,    # number of channels in the training images.
    ndf=64,  # size of feature maps in discriminator
)

noise = torch.randn(1, 100)
fake_img = netG(noise)
prediction = netD(fake_img)
```

### SRGAN

```python
import torch

from ganetic.srgan import Generator, Discriminator

img = torch.randn(1, 3, 64, 64)
gen = Generator(
    scale_factor=4, # scale factor for super resolution
    nci=3,          # number of channels in input image
    nco=3,          # number of channels in output image
    ngf=64,         # number of filters in the generator
    no_of_residual_blocks=5
)
disc = Discriminator(
    input_shape=(3, 256, 256),
    ndf=64,              # number of filters in the discriminator
    negative_slope=0.2,  # negative slope of leaky relu
)

HR_img = gen(img)
pred = disc(HR_img)
```

### Pix2Pix

```python
import torch

from ganetic.pix2pix import Discriminator, Generator

img = torch.randn(1, 3, 256, 256)
gen = Generator(
    nci=3,  # number of channels in input image
    nco=3,  # number of channels in output image
    ngf=64  # number of filters in the generator
)

disc = Discriminator(
    nci=3,  # number of channels in input image
    ndf=64  # number of filters in the discriminator
)

fake = gen(img)
pred = disc(img, fake)
```
### Conditional GANs

```python
import torch

from ganetic.cgan import Discriminator, Generator

gen = Generator(
    n_classes=10,
    nz=100,
    nc=3,
    ngf=64,
    out_size=64,
    activation='relu',
    last_activation='tanh'
)
disc = Discriminator(
    n_classes=10,
    nc=3,
    ndf=64,
    in_size=64,
    activation='LeakyReLU',
    last_activation='sigmoid'
)

z = torch.randn(64, 100)
label = torch.LongTensor(64).random_(0, 10)

print(gen(z, label).shape)
print(disc(gen(z, label), label).shape)
```

## Citations

```bibtex
@article{radford2015unsupervised,
  title={Unsupervised representation learning with deep convolutional generative adversarial networks},
  author={Radford, Alec and Metz, Luke and Chintala, Soumith},
  journal={arXiv preprint arXiv:1511.06434},
  year={2015}
}
```

```bibtex
@inproceedings{ledig2017photo,
  title={Photo-realistic single image super-resolution using a generative adversarial network},
  author={Ledig, Christian and Theis, Lucas and Husz{\'a}r, Ferenc and Caballero, Jose and Cunningham, Andrew and Acosta, Alejandro and Aitken, Andrew and Tejani, Alykhan and Totz, Johannes and Wang, Zehan and others},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={4681--4690},
  year={2017}
}
```

```bibtex
@inproceedings{isola2017image,
  title={Image-to-image translation with conditional adversarial networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1125--1134},
  year={2017}
}
```

```bibtex
@article{mirza2014conditional,
  title={Conditional generative adversarial nets},
  author={Mirza, Mehdi and Osindero, Simon},
  journal={arXiv preprint arXiv:1411.1784},
  year={2014}
}
```