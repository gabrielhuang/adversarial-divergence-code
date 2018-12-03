import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from scipy.ndimage.filters import gaussian_filter1d

if not os.path.exists('plots'):
    os.makedirs('plots')

smooth_window = 50
n_iterations = 20000

# For each subfolder get arguments
def load(path):
    with open('{}/args.json'.format(path), 'rb') as fp:
        arg = json.load(fp)

    # For each subfolder get stats
    print 'Opening', path
    with open('{}/stats.json'.format(path), 'rb') as fp:
        stat = json.load(fp)

    return arg, stat


def prepare_figure():
    plt.clf()
    plt.rcParams['text.latex.preamble'] = [r"\usepackage{lmodern}"]
    params = {'text.usetex': True,
              'font.size': 15,
              'font.family': 'lmodern',
              'text.latex.unicode': True,
              }
    plt.rcParams.update(params)



arg, stat = load('runs/vae')
print arg
print stat.keys()

# Smooth the stat
plt.figure()
prepare_figure()

smoothed = gaussian_filter1d(stat['Train/Accuracy'], smooth_window)
smoothed = smoothed[:min(len(smoothed), n_iterations)]
plt.plot(smoothed, label='Train(Test-25/Vae-Non25)')

smoothed = gaussian_filter1d(stat['Eval/NewCombination/Accuracy'], smooth_window)
smoothed = smoothed[:min(len(smoothed), n_iterations)]
plt.plot(smoothed, label='Eval(Test-25/Test-Non25)')

plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.ylim(0,1.1)
plt.title('No side task : Train(Test-25/Vae-Non25)')
plt.legend()
plt.savefig('plots/probe_train_vae.pdf')




arg, stat = load('runs/test')
print arg
print stat.keys()

# Smooth the stat
plt.figure()
prepare_figure()

smoothed = gaussian_filter1d(stat['Train/Accuracy'], smooth_window)
smoothed = smoothed[:min(len(smoothed), n_iterations)]
plt.plot(smoothed, label='Train(Test-25/Test-Non25)')

smoothed = gaussian_filter1d(stat['Eval/VaeNewCombination/Accuracy'], smooth_window)
smoothed = smoothed[:min(len(smoothed), n_iterations)]
plt.plot(smoothed, label='Eval(Test-25/Vae-Non25)')

smoothed = gaussian_filter1d(stat['Eval/GanNewCombination/Accuracy'], smooth_window)
smoothed = smoothed[:min(len(smoothed), n_iterations)]
plt.plot(smoothed, label='Eval(Test-25/Gan-Non25)')

plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.ylim(0,1.1)
plt.title(r'Side Task : Train(Test-25/Test-Non25)')
plt.legend()
plt.savefig('plots/probe_train_test.pdf')

