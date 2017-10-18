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
- Generate real numbers (real hyperplane)
- Generate one-hot encodings of integers (integer hyperplane)
- Generate concatenated images constructed from MNIST (visual hyperplane)

We can try different architectures for the visual one:
- End-to-end DCGAN/ConvVAE
- separate one-hot and image generation into two components

Evaluation Pipeline:
- Use a dataset, e.g.,  `dataset = HyperplaneCachedDataset(3, range(10), 4)` for combinations of 4 digits in [0..9] that sum up to 3.
- Split into train and test
- Evaluate precision and recall.
  - Precision: how many generated sum to the right number
  - Recall: how much of the held-out test set is covered.
- For computation of precision and recall, one will need to train a classifier to recognized generated digits, and then sum the recognized digits together.
