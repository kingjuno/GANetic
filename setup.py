import io

from setuptools import find_packages, setup

with io.open("README.md", "r", encoding="utf-8") as f:
    long_description = "\n" + f.read()

setup(
    name="GANetic",
    packages=find_packages(exclude=["docs", "tests*", "examples"]),
    version="0.0.9",
    license="MIT",
    description="A Collection of GANs - PyTorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Geo Jolly",
    author_email="geojollyc@gmail.com",
    url="https://github.com/kingjuno/GANetic",
    keywords=[
        "GAN",
        "Generative Adversarial Networks",
        "Deep Learning",
        "PyTorch",
        "GANetic",
        "Generative Artificial Intelligence",
    ],
    install_requires=[
        "torch>=1.10",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
