# DCGAN-MNIST

## Summary

This is a guided project from Coursera (Deep Learning with PyTorch : Generative Adversarial Network). 
A Deep Convolutional Generative Adversarial Network using PyTorch to generate handwritten digits, using dataset MNIST.

** The main difference is about the project structure, where is separate in responsibilities, without using a jupyter notebook.

![Net](docs/DCGAN.png)

### Network: Descriminator

```python
input: (bs, 1, 28, 28)

Conv2d(in_channel=1, out_channels=16, kernel_size=(3,3), stride=2)      # (bs, 16, 13, 13)
BatchNorm2d()                                                           # (bs, 16, 13, 13)
LeakyReLU()                                                             # (bs, 16, 13, 13)
    ...
Conv2d(in_channel=16, out_channels=32, kernel_size=(5,5), stride=2)     # (bs, 32, 5, 5)
BatchNorm2d()                                                           # (bs, 32, 5, 5)
LeakyReLU()                                                             # (bs, 32, 5, 5)
    ...
Conv2d(in_channel=32, out_channels=64, kernel_size=(5,5), stride=2)     # (bs, 64, 1, 1)
BatchNorm2d()                                                           # (bs, 64, 1, 1)
LeakyReLU()                                                             # (bs, 64, 1, 1)
    ...
Flatten()                                                               # (bs, 64)
Linear(in_features=64, out_features=1)                                  # (bs, 1)
```

### Network: Generator

```python
noise_dim = 64
input: (bs, noise_dim) --> (bs, channel, height, width) -> (bs, 64, 1, 1)
    ...
ConvTranspose2d(in_channel=z_dim, out_channel=256, kernel_size=(3,3), stride=2)             # (bs, 256, 3, 3)
BatchNorm2d()                                                                               # (bs, 256, 3, 3)
ReLU()                                                                                      # (bs, 256, 3, 3)
    ...
ConvTranspose2d(in_channel=256, out_channel=128, kernel_size=(4,4), stride=2)               # (bs, 128, 6, 6)
BatchNorm2d()                                                                               # (bs, 128, 6, 6)
ReLU()                                                                                      # (bs, 128, 6, 6)
    ...
ConvTranspose2d(in_channel=128, out_channel=64, kernel_size=(3,3), stride=2)                # (bs, 64, 13, 13)
BatchNorm2d()                                                                               # (bs, 64, 13, 13)
ReLU()                                                                                      # (bs, 64, 13, 13)
    ...
ConvTranspose2d(in_channel=64, out_channel=1, kernel_size=(4,4), stride=2)                  # (bs, 1, 28, 28)
Tanh()
```