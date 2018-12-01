import cPickle as pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import hyperplane_dataset
from tqdm import tqdm
import os
from collections import OrderedDict

import sys
sys.path.append('..')
from common import digits_sampler


# How many runs? (samples will be divided by that number)
n_runs = 1  # 5 - change to more
n_points = 5 # 30
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

# Result files
models = OrderedDict([
    ('GAN-SideTask', {'filename': 'results_wgan_side/digits.npy', 'color': 'xkcd:green', 'symbol':'-x'}),
    ('GAN', {'filename': 'results_wgan_noside/digits.npy', 'color': 'xkcd:blue', 'symbol':'-o'}),
    ('VAE', {'filename': 'results_vae/digits.npy', 'color': 'xkcd:red', 'symbol':'->'}),
])

# Load digits
for key, val in models.items():
    val['digits'] = np.load(val['filename'])

# This has to be the minimum
nb_samples = len(models['VAE']['digits'])

# generate baseline samples
flat_dataset = np.array(problem.positive).flatten()
samples_per_run = nb_samples / n_runs
print 'Going to do {} runs of {} samples'.format(n_runs, samples_per_run)

digits, freq = np.unique(flat_dataset, return_counts=True)
freq = freq / float(len(flat_dataset))

models['Baseline'] = {}
models['Baseline']['digits'] = np.random.choice(digits, size=(nb_samples, 5), p=freq)
models['Baseline']['color'] = 'xkcd:gray'
models['Baseline']['symbol'] = '-s'

# generate perfect samples
models['Perfect'] = {}
perfect_idx = np.random.randint(0, len(problem.positive), size=nb_samples)
models['Perfect']['digits'] = np.asarray(problem.positive)[perfect_idx]
models['Perfect']['color'] = 'xkcd:black'
models['Perfect']['symbol'] = '-d'

# Count sums
width = 1. / (len(models) + 1)
for key, val in models.items():
    val['sum'] = val['digits'].sum(axis=1)
    val['x'], val['height'] = np.unique(val['sum'], return_counts=True)
    val['height'] = val['height'] / float(len(val['sum'])) * 100


plt.clf()
plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 15,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)

for i, (model, val) in enumerate(models.items()):
    if model != 'Perfect':
        plt.bar(val['x'] + i*width, val['height'], width=width, alpha=0.6, color=val['color'], label=model)

plt.xlabel('sum of digits')
plt.ylabel('$\%$ frequency')
plt.legend(loc='best')
plt.xticks([0,10,20,25,30,40], ['0','10','20', r'\textbf{25}','30','40'])
plt.savefig('{}/sum_digits_dist.pdf'.format(output_dir))

################################
# RECALL
def get_recall(samples, dataset):
    intersection = [tuple(c) for c in samples if tuple(c) in dataset]
    unique = set(intersection)
    return len(unique) / float(len(dataset))

for model, val in models.items():
    print 'Getting recall for', model
    val['train_recalls'] = []

    all_train_recalls = []
    all_test_recalls = []

    for run in xrange(n_runs):
        print '\tRun {} of {}'.format(run, n_runs)
        train_recalls = []
        test_recalls = []
        samples = []

        for sample_ratio in tqdm(np.linspace(0, 1, n_points)):
            N = samples_per_run
            n_samples = max(1, min(int(sample_ratio*N), N))
            i = run*samples_per_run # starter index

            subset = val['digits'][i:i+n_samples]
            train_recall = get_recall(subset, train_positive_set)
            test_recall = get_recall(subset, test_positive_set)

            # accumulate
            samples.append(n_samples)
            train_recalls.append(train_recall)
            test_recalls.append(test_recall)

        all_train_recalls.append(train_recalls)
        all_test_recalls.append(test_recalls)

    val['train_recalls_mean'] = np.mean(all_train_recalls, axis=0)
    val['test_recalls_mean'] = np.mean(all_test_recalls, axis=0)


for model, val in models.items():
    print '{} recalls:'.format(model)
    print '\ttrain', val['train_recalls_mean']
    print '\ttest', val['test_recalls_mean']


# Make actual figure (full train test)
plt.figure()
plt.clf()
plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
params = {'text.usetex': True,
          'font.size': 12,
          'font.family': 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params)

samples_norm = np.asarray(samples) / float(len(problem.positive))

for model, val in models.items():
    if model in ['Baseline', 'Perfect']:
        plt.plot(samples_norm, val['test_recalls_mean'], val['symbol'], alpha=0.6, color=val['color'], label='{}'.format(model))
    else:
        plt.plot(samples_norm, val['test_recalls_mean'], val['symbol'], alpha=0.6, color=val['color'], label='{}'.format(model))
        plt.plot(samples_norm, val['train_recalls_mean'], val['symbol'], alpha=0.6, color=val['color'], linestyle='--')
        #plt.plot(samples_norm, val['test_recalls_mean'], val['symbol'], alpha=0.6, color=val['color'], label='{} (test)'.format(model))
        #plt.plot(samples_norm, val['train_recalls_mean'], val['symbol'], alpha=0.6, color=val['color'], label='{} (train)'.format(model), linestyle='--')

plt.xlabel('samples generated / number of total combinations')
plt.ylabel('recall')
plt.legend(loc='best')
plt.savefig('{}/recall.pdf'.format(output_dir))
