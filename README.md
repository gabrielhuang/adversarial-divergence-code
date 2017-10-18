# Code for paper Adversarial Divergence

## MILA-8

In this folder, we try to generate high-resolution images of the digit-8,
using a (de)convolutional VAE and a GAN. The goal is to see how important is the impact
of the following factors:
- VAE loss vs GAN loss
- Latent dimension
- Image resolution

## Visual-Hyperplane

We try to learn how to generate digits (integers) $d_1 ... d_k$ such that $d_1 + d_2 + ... + d_k = constant$.

There are three steps:
- Generate real numbers
- Generate one-hot encodings of integers
- Generate concatenated images constructed from MNIST
