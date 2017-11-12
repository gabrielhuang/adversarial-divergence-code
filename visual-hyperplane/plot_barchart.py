import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from hyperplane_dataset import get_full_train_test

full_dataset, train_dataset, test_dataset = get_full_train_test(25, range(10), 5, one_hot=False, validation=0.8, seed=1234)

vae_digits_filename = 'results-vae/digits.npy'
gan_digits_filename = 'results-gan/digits.npy'

vae_digits = np.load(vae_digits_filename)
gan_digits = np.load(gan_digits_filename)

vae_sum = vae_digits.sum(axis=1)
gan_sum = gan_digits.sum(axis=1)


# generate baseline samples
nb_samples = len(vae_digits)
flat_dataset = np.array(full_dataset.hyperplane_dataset.combinations).flatten()

digits, freq = np.unique(flat_dataset, return_counts=True)
freq = freq / float(len(flat_dataset))

base_digits = samples = np.random.choice(digits, size=(nb_samples, 5), p=freq)
baseline_sum = samples.sum(axis=1)

# generate perfect samples
perfect_idx = np.random.randint(0, len(full_dataset), size=nb_samples)
perfect_digits = np.asarray(full_dataset.hyperplane_dataset.combinations)[perfect_idx]

x_baseline, height_baseline = np.unique(baseline_sum, return_counts=True)
height_baseline = height_baseline/float(len(baseline_sum)) * 100
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
plt.xticks([0,10,20,25,30,40], ['0','10','20', r'\textbf{25}','30','40'])
plt.savefig('sum_digits_dist.pdf')

# RECALL
def get_recall(samples, dataset):
    intersection = [tuple(c) for c in samples if tuple(c) in dataset.set]
    unique = set(intersection)
    return len(unique) / float(len(dataset))

l_samples = []
l_vae_train = []
l_vae_test = []
l_gan_train = []
l_gan_test = []
l_base_train = []
l_base_test = []
l_perfect_train = []
l_perfect_test = []
for sample_ratio in np.linspace(0, 1, 30):
    N = min(len(vae_digits), len(gan_digits))
    n_samples = max(1, min(int(sample_ratio*N), N))

    #recall_vae_full = get_recall(vae_digits[:n_samples], full_dataset)
    recall_vae_train = get_recall(vae_digits[:n_samples], train_dataset)
    recall_vae_test = get_recall(vae_digits[:n_samples], test_dataset)

    #recall_gan_full = get_recall(gan_digits[:n_samples], full_dataset)
    recall_gan_train = get_recall(gan_digits[:n_samples], train_dataset)
    recall_gan_test = get_recall(gan_digits[:n_samples], test_dataset)

    #recall_base_full = get_recall(base_digits[:n_samples], full_dataset)
    recall_base_train = get_recall(base_digits[:n_samples], train_dataset)
    recall_base_test = get_recall(base_digits[:n_samples], test_dataset)

    #recall_perfect_full = get_recall(perfect_digits[:n_samples], full_dataset)
    recall_perfect_train = get_recall(perfect_digits[:n_samples], train_dataset)
    recall_perfect_test = get_recall(perfect_digits[:n_samples], test_dataset)

    # accumulate
    l_samples.append(n_samples)
    l_vae_train.append(recall_vae_train)
    l_vae_test.append(recall_vae_test)
    l_gan_train.append(recall_gan_train)
    l_gan_test.append(recall_gan_test)
    l_base_train.append(recall_base_train)
    l_base_test.append(recall_base_test)
    l_perfect_train.append(recall_perfect_train)
    l_perfect_test.append(recall_perfect_test)
l_samples = np.asarray(l_samples)

print 'n_samples', n_samples

print 'Baseline recalls:'
#print '\tfull', recall_base_full
print '\ttrain', recall_base_train
print '\ttest', recall_base_test

print 'VAE recalls:'
#print '\tfull', recall_vae_full
print '\ttrain', recall_vae_train
print '\ttest', recall_vae_test

print 'GAN recalls:'
#print '\tfull', recall_gan_full
print '\ttrain', recall_gan_train
print '\ttest', recall_gan_test

print 'Perfect recalls:'
#print '\tfull', recall_perfect_full
print '\ttrain', recall_perfect_train
print '\ttest', recall_perfect_test

# Make actual figure
plt.figure()
plt.clf()
plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 10,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)

plt.plot(l_samples, l_gan_test, '-o', alpha=0.6, color='green', label='GAN (test)')
plt.plot(l_samples, l_gan_train, '-o', alpha=0.6, color='green', label='GAN (train)', linestyle='--')

plt.plot(l_samples, l_vae_test, '-o', alpha=0.6, color='red', label='VAE (test)')
plt.plot(l_samples, l_vae_train, '-o', alpha=0.6, color='red', label='VAE (train)', linestyle='--')

plt.plot(l_samples, l_base_test, '-o', alpha=0.6, color='gray', label='Indep. baseline (test)')
plt.plot(l_samples, l_base_train, '-o', alpha=0.6, color='gray', label='Indep. baseline (train)', linestyle='--')

plt.plot(l_samples, l_perfect_test, '-o', alpha=0.6, color='blue', label='Perfect (test)')
plt.plot(l_samples, l_perfect_train, '-o', alpha=0.6, color='blue', label='Perfect (train)', linestyle='--')

plt.xlabel('samples generated')
plt.ylabel('recall')
plt.legend(loc='best')
plt.savefig('recall.pdf')

# Make actual figure
plt.figure()
plt.clf()
plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 10,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)

train_ratio = l_samples / float(len(train_dataset))
test_ratio = l_samples / float(len(test_dataset))
train_alpha = 0.3
test_alpha = 0.9
plt.plot(test_ratio, l_gan_test, '-', alpha=test_alpha, color='green', label='GAN (test)')
plt.plot(train_ratio, l_gan_train, '-', alpha=train_alpha, color='green', label='GAN (train)', linestyle='--')

plt.plot(test_ratio, l_vae_test, '-', alpha=test_alpha, color='red', label='VAE (test)')
plt.plot(train_ratio, l_vae_train, '-', alpha=train_alpha, color='red', label='VAE (train)', linestyle='--')

plt.plot(test_ratio, l_base_test, '-', alpha=test_alpha, color='gray', label='Indep. baseline (test)')
plt.plot(train_ratio, l_base_train, '-', alpha=train_alpha, color='gray', label='Indep. baseline (train)', linestyle='--')

plt.plot(test_ratio, l_perfect_test, '-', alpha=test_alpha, color='blue', label='Perfect (test)')
plt.plot(train_ratio, l_perfect_train, '-', alpha=train_alpha, color='blue', label='Perfect (train)', linestyle='--')

plt.xlabel('samples generated / size(target set)')
plt.ylabel('recall')
plt.xlim([0,40])
plt.legend(loc='best')
plt.savefig('recall_relative.pdf')

