import cPickle as pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import hyperplane_dataset
from tqdm import tqdm
import os

import sys
sys.path.append('..')
from common import digits_sampler


# How many runs? (samples will be divided by that number)
n_runs = 1  # 5 - change to more
n_points = 3 # 30
problem_file = '../end2end/sum_25.pkl'
output_dir = 'plots'

# Load problem
print 'Loading problem {}'.format(problem_file)
with open(problem_file, 'rb') as fp:
    problem = pickle.load(fp)
print problem
# Compute sets
positive_set = set(tuple(x) for x in problem.positive)
train_positive_set = set(tuple(x) for x in problem.train_positive)
test_positive_set = set(tuple(x) for x in problem.test_positive)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


gan_digits_filename = 'results_wgan_side/digits.npy'
vae_digits_filename = 'results_wgan_noside/digits.npy'

vae_digits = np.load(vae_digits_filename)
gan_digits = np.load(gan_digits_filename)

#vae_digits = vae_digits[:100000]
#gan_digits = gan_digits[:100000]

vae_sum = vae_digits.sum(axis=1)
gan_sum = gan_digits.sum(axis=1)


# generate baseline samples
nb_samples = len(vae_digits)
flat_dataset = np.array(problem.positive).flatten()
samples_per_run = nb_samples / n_runs
print 'Going to do {} old_runs_2 of {} samples'.format(n_runs, samples_per_run)

digits, freq = np.unique(flat_dataset, return_counts=True)
freq = freq / float(len(flat_dataset))

base_digits = samples = np.random.choice(digits, size=(nb_samples, 5), p=freq)
baseline_sum = samples.sum(axis=1)

# generate perfect samples
perfect_idx = np.random.randint(0, len(problem.positive), size=nb_samples)
perfect_digits = np.asarray(problem.positive)[perfect_idx]

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
plt.savefig('{}/sum_digits_dist.pdf'.format(output_dir))

# RECALL
def get_recall(samples, dataset):
    intersection = [tuple(c) for c in samples if tuple(c) in dataset]
    unique = set(intersection)
    return len(unique) / float(len(dataset))

l_vae_train = []
l_vae_test = []
l_gan_train = []
l_gan_test = []
l_base_train = []
l_base_test = []
l_perfect_train = []
l_perfect_test = []
for run in xrange(n_runs):
    print 'Run {} of {}'.format(run, n_runs)

    l_samples = []
    r_vae_train = []
    r_vae_test = []
    r_gan_train = []
    r_gan_test = []
    r_base_train = []
    r_base_test = []
    r_perfect_train = []
    r_perfect_test = []
    for sample_ratio in tqdm(np.linspace(0, 1, n_points)):
        N = samples_per_run
        n_samples = max(1, min(int(sample_ratio*N), N))
        i = run*samples_per_run # starter index

        #recall_vae_full = get_recall(vae_digits[i:i+n_samples], full_dataset)
        recall_vae_train = get_recall(vae_digits[i:i+n_samples], train_positive_set)
        recall_vae_test = get_recall(vae_digits[i:i+n_samples], test_positive_set)

        #recall_gan_full = get_recall(gan_digits[i:i+n_samples], full_dataset)
        recall_gan_train = get_recall(gan_digits[i:i+n_samples], train_positive_set)
        recall_gan_test = get_recall(gan_digits[i:i+n_samples], test_positive_set)

        #recall_base_full = get_recall(base_digits[i:i+n_samples], full_dataset)
        recall_base_train = get_recall(base_digits[i:i+n_samples], train_positive_set)
        recall_base_test = get_recall(base_digits[i:i+n_samples], test_positive_set)

        #recall_perfect_full = get_recall(perfect_digits[i:i+n_samples], full_dataset)
        recall_perfect_train = get_recall(perfect_digits[i:i+n_samples], train_positive_set)
        recall_perfect_test = get_recall(perfect_digits[i:i+n_samples], test_positive_set)

        # accumulate
        l_samples.append(n_samples)
        r_vae_train.append(recall_vae_train)
        r_vae_test.append(recall_vae_test)
        r_gan_train.append(recall_gan_train)
        r_gan_test.append(recall_gan_test)
        r_base_train.append(recall_base_train)
        r_base_test.append(recall_base_test)
        r_perfect_train.append(recall_perfect_train)
        r_perfect_test.append(recall_perfect_test)

    l_samples = np.asarray(l_samples)

    l_vae_train.append(r_vae_train)
    l_vae_test.append(r_vae_test)
    l_gan_train.append(r_gan_train)
    l_gan_test.append(r_gan_test)
    l_base_train.append(r_base_train)
    l_base_test.append(r_base_test)
    l_perfect_train.append(r_perfect_train)
    l_perfect_test.append(r_perfect_test)

    print 'n_samples', n_samples
l_vae_train = np.mean(l_vae_train, axis=0)
l_vae_test = np.mean(l_vae_test, axis=0)
l_gan_train = np.mean(l_gan_train, axis=0)
l_gan_test = np.mean(l_gan_test, axis=0)
l_base_train = np.mean(l_base_train, axis=0)
l_base_test = np.mean(l_base_test, axis=0)
l_perfect_train = np.mean(l_perfect_train, axis=0)
l_perfect_test = np.mean(l_perfect_test, axis=0)

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


# Make actual figure (full train test)
plt.figure()
plt.clf()
plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 15,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)

l_samples_norm = l_samples / float(len(problem.positive))

plt.plot(l_samples_norm, l_gan_test, '-o', alpha=0.6, color='green', label='GAN (test)')
plt.plot(l_samples_norm, l_gan_train, '-o', alpha=0.6, color='green', label='GAN (train)', linestyle='--')

plt.plot(l_samples_norm, l_vae_test, '->', alpha=0.6, color='red', label='VAE (test)')
plt.plot(l_samples_norm, l_vae_train, '->', alpha=0.6, color='red', label='VAE (train)', linestyle='--')

plt.plot(l_samples_norm, l_base_test, '-s', alpha=0.6, color='gray', label='Indep. Baseline (test)')
plt.plot(l_samples_norm, l_base_train, '-s', alpha=0.6, color='gray', label='Indep. Baseline (train)', linestyle='--')

plt.plot(l_samples_norm, l_perfect_test, '-d', alpha=0.6, color='blue', label='Perfect (test)')
plt.plot(l_samples_norm, l_perfect_train, '-d', alpha=0.6, color='blue', label='Perfect (train)', linestyle='--')

plt.xlabel('samples generated / number of total combinations')
plt.ylabel('recall')
plt.legend(loc='best')
plt.savefig('{}/recall.pdf'.format(output_dir))



# Make actual figure (merged train_test)
plt.figure()
plt.clf()
plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 15,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)

l_samples_norm = l_samples / float(len(problem.positive))

plt.plot(l_samples_norm, l_gan_test, '-o', alpha=0.6, color='green', label='GAN (test)')
plt.plot(l_samples_norm, l_gan_train, '-o', alpha=0.6, color='green', label='GAN (train)', linestyle='--')

plt.plot(l_samples_norm, l_vae_test, '->', alpha=0.6, color='red', label='VAE (test)')
plt.plot(l_samples_norm, l_vae_train, '->', alpha=0.6, color='red', label='VAE (train)', linestyle='--')

plt.plot(l_samples_norm, l_base_train, '-s', alpha=0.6, color='gray', label='Indep. Baseline')

plt.plot(l_samples_norm, l_perfect_train, '-d', alpha=0.6, color='blue', label='Perfect')

plt.xlabel('samples generated / number of total combinations')
plt.ylabel('recall')
plt.legend(loc='best')
plt.savefig('{}/recall_merged.pdf'.format(output_dir))

