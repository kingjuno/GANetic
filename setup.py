from setuptools import setup, find_packages

setup(
    name='GAN-Zoo',
    packages=find_packages(exclude=['docs', 'tests*', 'examples']),
    version='0.1.0',
    license='MIT',
    description='A Collection of GANs - PyTorch',
    author='Geo Jolly',
    author_email='geojollyc@gmail.com',
    url='https://github.com/kingjuno/GAN-Zoo',
    keywords=[
        'GAN',
        'Generative Adversarial Networks',
        'Deep Learning',
        'PyTorch', 
        'GAN-Zoo'
    ],
    install_requires=[
        'torch>=1.13.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ]
)
