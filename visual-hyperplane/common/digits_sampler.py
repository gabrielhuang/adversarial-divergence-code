import numpy as np
import torch
from torchvision import datasets, transforms


#############################
# Combination to Visual
def combination_to_visual(combination, visual_samplers):
    '''
    Sample a visual representation for a symbolic combination

    Parameters
    ----------
    combination: np.ndarray (part~5) combination to encode
    visual_samplers: dict of 10 callable objects which return a random corresponding (28, 28) digit

    Returns
    -------
    visual_combination: torch.Tensor (part~5, height~28, width~28)
    '''
    # accumulate along channel dimension
    x = []
    for c in combination:
        sample = visual_samplers[c]()
        x.append(sample)
    visual_combination = torch.cat(x, 1)
    return visual_combination

# Load only one digit from mnist
def load_one_mnist_digit(digit, train, debug=False):
    test_digits = []
    for i, (data, label) in enumerate(datasets.MNIST('../data', train=train, download=True, transform=transforms.ToTensor())):
        if digit == -1 or label.item() == digit:
            test_digits.append((data, label))
            if debug and len(test_digits) > 10:
                print 'WARNING THIS IS DEBUG MODE'
                break
    return test_digits


def make_infinite(iterable):
    while True:
        for x in iterable:
            yield x


# Standard MNIST digit
def load_full_mnist_digits(batch_size):
    train_iter = make_infinite(torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
                       batch_size=batch_size))
    test_iter = make_infinite(torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True,
                       transform=transforms.ToTensor()),
                       batch_size=batch_size))
    return train_iter, test_iter



class DatasetVisualSampler(object):
    def __init__(self, digit_iter):
        self.digit_iter = digit_iter

    def __call__(self):
            return self.digit_iter.next()[0]


class VaeVisualSampler(object):
    def __init__(self, vae, batch_size=64):
        self.idx = batch_size
        self.batch_size = batch_size
        self.vae = vae

    def __call__(self):  # return one sample
        if self.idx >= self.batch_size:
            # Regenerate new
            self.cache = self.vae.generate(self.batch_size)
            self.idx = 0
        sample = self.cache[self.idx][None, ...]  # preserve dims
        self.idx += 1
        return sample.detach()


def sample_visual_combination(symbolic, visual, combination_batch_size):
    samples = []
    for j in xrange(combination_batch_size):
        # Sample combination from symbolic
        combination_idx = np.random.choice(len(symbolic))
        combination = symbolic[combination_idx]

        # Make visual
        samples.append(combination_to_visual(combination, visual))
    samples = torch.cat(samples, 0)
    return samples
