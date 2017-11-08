import matplotlib.pyplot as plt
import numpy as np
from hyperplane_dataset import get_full_train_test

full_dataset, train_dataset, test_dataset = get_full_train_test(25, range(10), 5, one_hot=False, validation=0.8, seed=1234)

# generate baseline samples
nb_samples = 70000
flat_dataset = np.array(full_dataset.hyperplane_dataset.combinations).flatten()

digits, freq = np.unique(flat_dataset, return_counts=True)
freq = freq / float(len(flat_dataset))

samples = np.random.choice(digits, size=(nb_samples, 5), p=freq)
baseline_sum = samples.sum(axis=1)

x_baseline, height_baseline = np.unique(baseline_sum, return_counts=True)
height_baseline = height_baseline/float(len(baseline_sum)) * 100

vae_digits_filename = 'results-vae/digits.npy'
gan_digits_filename = 'results-gan/digits.npy'

vae_digits = np.load(vae_digits_filename)
gan_digits = np.load(gan_digits_filename)

vae_sum = vae_digits.sum(axis=1)
gan_sum = gan_digits.sum(axis=1)

x_gan, height_gan = np.unique(gan_sum, return_counts=True)
height_gan = height_gan/float(len(gan_sum)) * 100

x_vae, height_vae = np.unique(vae_sum, return_counts=True)
height_vae = height_vae/float(len(vae_sum)) * 100

width = 0.30

plt.clf()
plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 15,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)

plt.bar(x_gan, height_gan, width=width, alpha=0.6, color='green', label='GAN')
plt.bar(x_vae + width, height_vae, width=width, alpha=0.6, color='red', label='VAE')
plt.bar(x_baseline + 2*width, height_baseline, width=width, alpha=0.6, color='gray', label='Indep. Baseline')
plt.xlabel('sum of digits')
plt.ylabel('$\%$ frequency')
plt.legend(loc='best')
plt.savefig('sum_digits_dist.pdf')

# RECALL
def get_recall(samples, dataset):
    intersection = [tuple(c) for c in samples if tuple(c) in dataset.set]
    unique = set(intersection)
    return len(unique) / float(len(dataset))

recall_vae_full = get_recall(vae_digits, full_dataset)
recall_vae_train = get_recall(vae_digits, train_dataset)
recall_vae_test = get_recall(vae_digits, test_dataset)

recall_gan_full = get_recall(gan_digits, full_dataset)
recall_gan_train = get_recall(gan_digits, train_dataset)
recall_gan_test = get_recall(gan_digits, test_dataset)

print 'VAE recalls:'
print '\tfull', recall_vae_full
print '\ttrain', recall_vae_train
print '\ttest', recall_vae_test

print 'GAN recalls:'
print '\tfull', recall_gan_full
print '\ttrain', recall_gan_train
print '\ttest', recall_gan_test
