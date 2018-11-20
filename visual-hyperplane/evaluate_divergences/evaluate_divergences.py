import sys
sys.path.append('..')

import torch
import numpy as np
import time
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

from common.models import VAE, Discriminator2, Discriminator4
from common.problems import get_problem
from common import digits

###############################

# All parameters go here
class Parameters(object):
    pass


class Summary(object):
    def __init__(self):
        self.logs = {}

    def log(self, epoch, name, value):
        self.logs.setdefault(name, {})
        self.logs[name][epoch] = value

    def sorted(self):
        sorted_logs = {}
        for log in self.logs:
            sorted_logs[log] = sorted(self.logs[log].items())
        return sorted_logs

    def get_arrays(self):
        arrays = {}
        for log in self.logs:
            arrays[log] = zip(*sorted(self.logs[log].items()))[1]
        return arrays

    def dump_arrays(self, fname):
        arrays = self.get_arrays()
        with open(fname, 'wb') as fp:
            json.dump(arrays, fp, indent=4)

    def print_summary(self, n_avg=50):
        sorted_logs = self.sorted()
        print 'Summary'
        for log in sorted_logs:
            tail = sorted_logs[log]
            tail = tail[-min(len(tail), n_avg):]
            val = dict(tail).values()
            print '\t{}: {:.4f} +/- {:.4f}'.format(log, np.mean(val), np.std(val))

p = Parameters()

# Train/Test Ratio
p.TRAIN_RATIO = 0.5

#p.DEBUG_TEST = False
p.DEBUG_TEST = True
#print 'Warning DEBUG'

###### Surrogate task: Classify individual digits
# if negative, do not learn to classify.
#p.LEARN_TO_CLASSIFY = 0.0
#p.LEARN_TO_CLASSIFY = 0.1
#p.LEARN_TO_CLASSIFY = 0.
# Batch-size for individual digits
p.DIGIT_BATCH_SIZE = 64

#p.ONLY_CLASSIFY = False  # no training other than classify (for debbuging)

###### GAN training
# Total training iterations
p.ITERATIONS = 10000

# Gradient penalty
p.PENALTY = 0.  # 10.
p.LR = 1e-3  # Adam learning rate

# Batch-size for visual combinations
p.COMBINATION_BATCH_SIZE = 12

###### Architecture
p.ARCHITECTURE = 'Discriminator2'
#p.ARCHITECTURE = 'Discriminator4'


#############################
# Create output directory
run_dir = 'runs/{}'.format(time.strftime("run-%Y.%m.%d-%H.%M.%S"))
print 'Creating', run_dir
os.makedirs(run_dir)

# Dump parameters
with open('{}/args.json'.format(run_dir),'wb') as fp:
    json.dump(vars(p), fp, indent=4)
with open('{}.DIVERGENCE'.format(run_dir), 'wb') as fp:
    pass


#############################
def make_infinite(iterable):
    while True:
        for x in iterable:
            yield x


def compute_gradient_penalty(netD, data):
    # Dunnow how to do this better with detach
    data = torch.autograd.Variable(data.detach(), requires_grad=True)
    outputs = netD(data)

    gradients = torch.autograd.grad(outputs=outputs,
                                    inputs=data,
                                    grad_outputs=torch.ones(outputs.size()),
                                    create_graph=True,
                                    retain_graph=True,
                                    only_inputs=True)[0]

    # Careful with the dimensions!! The input is multidimensional
    #old_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    #new_gradient_penalty = torch.max((gradients**2).sum() - 1., torch.zeros(1))
    relaxed_gradient_penalty = (gradients**2).sum() / float(len(data))
    return relaxed_gradient_penalty



def export_pdf(fname, stats, smooth_window=21, fontsize=12):
    stats = vars(stats)
    width = 3
    plt.figure()
    plt.tight_layout()
    for i, (key, value) in enumerate(stats.items()):
        # Smooth
        smoothed = gaussian_filter1d(value, smooth_window)

        plt.subplot(width, width, i + 1)
        # plt.plot(value)
        plt.plot(smoothed)
        plt.xlabel('Iterations', fontsize=fontsize)
        plt.ylabel(key, fontsize=fontsize)

    plt.savefig(fname)

##################
# Load all VAES
print 'Loading VAEs'
vaes = {}
# EPOCH = 80
EPOCH = 70
for i in xrange(10):
    vae = VAE()
    state_dict = torch.load('epoch_{}/digit_{}_epoch_{}.pth'.format(EPOCH, i, EPOCH))
    vae.load_state_dict(state_dict)
    vaes[i] = vae
#####################################################
print 'Loading MNIST digits'

# Load individual MNIST digits
digit_test_iter = {}
for i in xrange(10):
    print 'Loading digit', i
    test_digit = digits.load_one_mnist_digit(i, train=False, debug=p.DEBUG_TEST)
    digit_test_iter[i] = make_infinite(
        torch.utils.data.DataLoader(test_digit, batch_size=1, shuffle=True))
#####################################################

# Load all MNIST digits
train_iter, test_iter = digits.load_full_mnist_digits(batch_size=p.DIGIT_BATCH_SIZE)

#####################################################
vae_visual_samplers = [digits.ModelVisualSampler(vaes[i]) for i in xrange(10)]
test_visual_samplers = [digits.DatasetVisualSampler(digit_test_iter[i]) for i in xrange(10)]


# Load the two problems
sum_25 = get_problem('sum_25', 'int', train_ratio=p.TRAIN_RATIO)
non_25 = get_problem('non_25', 'int', train_ratio=p.TRAIN_RATIO)

# Training Visual Distributions
train_p_visual = test_visual_samplers
train_q_visual = test_visual_samplers
train_p_symbolic = sum_25.train_positive
train_q_symbolic = non_25.train_positive

eval_pairs = {}

# Eval (same visual, different combinations)
eval_pairs['NewCombination'] = {
    'p_visual': test_visual_samplers,
    'q_visual': test_visual_samplers,
    'p_symbol': sum_25.test_positive,
    'q_symbol': non_25.test_positive,
}

# Eval (different visual, same combination)
eval_pairs['NewVisual'] = {
    'p_visual': test_visual_samplers,
    'q_visual': vae_visual_samplers,
    'p_symbol': sum_25.train_positive,
    'q_symbol': non_25.train_positive,
}

# Eval (different visual, different combination)
eval_pairs['NewVisualNewCombination'] = {
    'p_visual': test_visual_samplers,
    'q_visual': vae_visual_samplers,
    'p_symbol': sum_25.test_positive,
    'q_symbol': non_25.test_positive,
}


##########################################
# Create discriminator
DiscriminatorClass = eval(p.ARCHITECTURE)
discriminator = DiscriminatorClass()
print discriminator
optimizer = torch.optim.Adam(discriminator.parameters(), lr=p.LR)

##########################################
# Actual training code

criterion = torch.nn.BCELoss()
nll_loss = torch.nn.NLLLoss()

summary = Summary()

try:
    for iteration in xrange(p.ITERATIONS):
        ####################################
        # Sample visual combination
        p_digit_images = sample_visual_combination(train_p_symbolic, train_p_visual, p.COMBINATION_BATCH_SIZE)
        q_digit_images = sample_visual_combination(train_q_symbolic, train_q_visual, p.COMBINATION_BATCH_SIZE)

        ####################################
        #  Train discriminator

        # Compute output
        p_out = discriminator(p_digit_images)
        q_out = discriminator(q_digit_images)

        p_target = torch.ones(len(p_out)) # REAL is ONE, FAKE is ZERO
        q_target = torch.zeros(len(q_out)) # REAL is ONE, FAKE is ZERO

        # Compute loss
        p_loss = criterion(p_out, p_target)
        q_loss = criterion(q_out, q_target)
        classifier_loss = 0.5 * (p_loss + q_loss)
        classifier_accuracy = 0.5 * ((p_out>0.5).float().mean() + (q_out<=0.5).float().mean())

        # Compute penalty
        p_penalty = compute_gradient_penalty(discriminator, p_digit_images)
        q_penalty = compute_gradient_penalty(discriminator, q_digit_images)
        penalty_loss = p.PENALTY * (p_penalty + q_penalty)

        total_loss = classifier_loss + penalty_loss



        # Learn to classify?
        #if p.LEARN_TO_CLASSIFY >= 0.:
        #    data, target = train_iter.next()
        #    output = discriminator.classify_digit(data)
        #    digit_loss = nll_loss(output, target)
        #    s.digit_losses.append(digit_loss.item())
        #    digit_accuracy = (output.argmax(1) == target).float().mean()
        #    s.digit_accuracies.append(digit_accuracy.item())

        #    if p.ONLY_CLASSIFY:
        #        total_loss = digit_loss
        #    else:
        #        total_loss = total_loss + p.LEARN_TO_CLASSIFY * digit_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Log
        summary.log(iteration, 'Train/Loss', classifier_loss.item())
        summary.log(iteration, 'Train/Accuracy', classifier_accuracy.item())

        ######################################
        # Evaluate if it can distinguish constraint when using perfect visual samples
        # difference with training is model_visual_samplers are replaced with target_visual_samplers
        #eval_target_visual = sample_visual_combination(target_symbolic_samplers, target_visual_samplers, p.COMBINATION_BATCH_SIZE)
        #eval_model_visual = sample_visual_combination(model_symbolic_samplers, target_visual_samplers, p.COMBINATION_BATCH_SIZE)

        for name, eval_pair in eval_pairs.items():

            # Evaluate when it can distinguish with imperfect samples
            p_digit_images = sample_visual_combination(eval_pair['p_symbol'], eval_pair['p_visual'], p.COMBINATION_BATCH_SIZE)
            q_digit_images = sample_visual_combination(eval_pair['q_symbol'], eval_pair['q_visual'], p.COMBINATION_BATCH_SIZE)

            # Compute output
            p_out = discriminator(p_digit_images)
            q_out = discriminator(q_digit_images)

            # Compute loss
            p_loss = criterion(p_out, p_target)
            q_loss = criterion(q_out, q_target)
            classifier_loss = 0.5 * (p_loss + q_loss)
            classifier_accuracy = 0.5 * ((p_out>0.5).float().mean() + (q_out<=0.5).float().mean())

            # Compute calibrated accuracy
            thresholds = torch.linspace(0, 1, 1000)
            calibrated_accuracy = 0.5 * torch.max((p_out > thresholds).float().mean(0)
                                                       + (q_out <= thresholds).float().mean(0))

            # Log
            summary.log(iteration, 'Eval/{}/Loss'.format(name), classifier_loss.item())
            summary.log(iteration, 'Eval/{}/Accuracy'.format(name), classifier_accuracy.item())
            #summary.log(iteration, 'Eval/{}/CalibratedAccuracy'.format(name), calibrated_accuracy.item())
        ######################################
        # Evaluate classification accuracy on model distribution
        #target = torch.LongTensor(list(range(10))*6)
        #individual_visual = combination_to_visual(target, model_visual_samplers)
        #individual_visual = individual_visual.view(-1, 1, 28, 28)
        #output = discriminator.classify_digit(individual_visual)
        #digit_loss = nll_loss(output, target)
        #s.model_digit_losses.append(digit_loss.item())
        #digit_accuracy = (output.argmax(1) == target).float().mean()
        #s.model_digit_accuracies.append(digit_accuracy.item())

        ######################################
        # Log
        if iteration % 50 == 0:
            print '\nIteration', iteration
            summary.print_summary()

            # Dump data
            summary.dump_arrays('{}/stats.json'.format(run_dir))


except KeyboardInterrupt:
    print 'interrupted'
