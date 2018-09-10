import sys
sys.path.append('..')
import torch
from torchvision import datasets, transforms
from models import VAE, loss_function, Discriminator1, Discriminator2, Discriminator3, Discriminator4
from problems import get_problem
import numpy as np
import time
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import math

###############################

# All parameters go here
class Parameters(object):
    pass


class Stats(object):
    pass


p = Parameters()


####### Which distributions
# Visual conditional models
p.TARGET_VISUAL = 'test'
#p.MODEL_VISUAL = 'test'
p.MODEL_VISUAL = 'vae'
p.DEBUG_TEST = False
#p.DEBUG_TEST = True

# Symbolic conditional models
p.TARGET_SYMBOL = 'sum_25'
p.MODEL_SYMBOL = 'uniform'

###### Surrogate task: Classify individual digits
# if negative, do not learn to classify.
#p.LEARN_TO_CLASSIFY = 0.0
#p.LEARN_TO_CLASSIFY = 0.1
p.LEARN_TO_CLASSIFY = 1.
# Batch-size for individual digits
p.DIGIT_BATCH_SIZE = 64

p.ONLY_CLASSIFY = False  # no training other than classify (for debbuging)

###### GAN training
# Total training iterations
p.ITERATIONS = 10000

# Gradient penalty
p.PENALTY = 0.  # 10.
p.LR = 1e-3  # Adam learning rate

# Batch-size for visual combinations
p.COMBINATION_BATCH_SIZE = 12

###### Architecture
#p.ARCHITECTURE = 'Discriminator2'
p.ARCHITECTURE = 'Discriminator4'


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


class DatasetVisualSampler(object):
    def __init__(self, digit_iter):
        self.digit_iter = digit_iter

    def __call__(self):
            return self.digit_iter.next()[0]


class ModelVisualSampler(object):
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
# Load some problems
uniform = get_problem('uniform', 'int', train_ratio=1.)
sum_25 = get_problem('sum_25', 'int', train_ratio=1.)
increasing = get_problem('increasing', 'int', train_ratio=1.)
symmetric = get_problem('symmetric', 'int', train_ratio=1.)
even = get_problem('even', 'int', train_ratio=1.)

#####################################################
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
    test_digit = load_one_mnist_digit(i, train=False, debug=p.DEBUG_TEST)
    digit_test_iter[i] = make_infinite(
        torch.utils.data.DataLoader(test_digit, batch_size=1, shuffle=True))
#####################################################

# Load all MNIST digits
train_iter, test_iter = load_full_mnist_digits(batch_size=p.DIGIT_BATCH_SIZE)

#####################################################
# Create visual samplers
vae_visual_samplers = [ModelVisualSampler(vaes[i]) for i in xrange(10)]
test_visual_samplers = [DatasetVisualSampler(digit_test_iter[i]) for i in xrange(10)]

if p.MODEL_VISUAL == 'test':
    model_visual_samplers = test_visual_samplers
elif p.MODEL_VISUAL == 'vae':
    model_visual_samplers = vae_visual_samplers
else:
    raise ValueError()

if p.TARGET_VISUAL == 'test':
    target_visual_samplers = test_visual_samplers
else:
    raise ValueError()


##########################################
# Create symbolic samplers

# Pick target distribution
target_symbolic_samplers = get_problem(p.TARGET_SYMBOL, 'int', train_ratio=1.).train_positive

# Pick model joint distribution
model_symbolic_samplers = get_problem(p.MODEL_SYMBOL, 'int', train_ratio=1.).train_positive

##########################################
# Create discriminator
DiscriminatorClass = eval(p.ARCHITECTURE)
discriminator = DiscriminatorClass()
print discriminator
optimizer = torch.optim.Adam(discriminator.parameters(), lr=p.LR)

##########################################
# Actual training code

criterion = torch.nn.BCELoss()

s = Stats()
s.classifier_losses = []
s.classifier_accuracies = []
s.penalty_losses = []
s.total_losses = []
s.eval_classifier_accuracies = []
s.eval_calibrated_accuracies = []
s.digit_losses = []
s.digit_accuracies = []
s.eval_target_outputs = []
s.eval_model_outputs = []

nll_loss = torch.nn.NLLLoss()

def summarize(u, suffix=50):
    v = u[-min(suffix, len(u)):]
    return '{:.3f} +/- {:.3f}'.format(np.mean(v), np.std(v)/np.sqrt(len(v)))

try:
    for iteration in xrange(p.ITERATIONS):
        ####################################
        # Sample visual combination
        target_visual = sample_visual_combination(target_symbolic_samplers, target_visual_samplers, p.COMBINATION_BATCH_SIZE)
        model_visual = sample_visual_combination(model_symbolic_samplers, model_visual_samplers, p.COMBINATION_BATCH_SIZE)

        ####################################
        #  Train discriminator

        # Compute output
        target_output = discriminator(target_visual)
        model_output = discriminator(model_visual)

        target_target = torch.ones(len(target_output)) # REAL is ONE, FAKE is ZERO
        model_target = torch.zeros(len(model_output)) # REAL is ONE, FAKE is ZERO

        # Compute loss
        target_score = criterion(target_output, target_target)
        model_score = criterion(model_output, model_target)
        classifier_loss = 0.5 * (target_score + model_score)
        classifier_accuracy = 0.5 * ((target_output>0.5).float().mean() + (model_output<=0.5).float().mean())

        # Compute penalty
        target_penalty = compute_gradient_penalty(discriminator, target_visual)
        model_penalty = compute_gradient_penalty(discriminator, model_visual)
        penalty_loss = p.PENALTY * (target_penalty + model_penalty)

        total_loss = classifier_loss + penalty_loss

        # Learn to classify?
        if p.LEARN_TO_CLASSIFY >= 0.:
            data, target = train_iter.next()
            output = discriminator.classify_digit(data)
            digit_loss = nll_loss(output, target)
            s.digit_losses.append(digit_loss.item())
            digit_accuracy = (output.argmax(1) == target).float().mean()
            s.digit_accuracies.append(digit_accuracy.item())

            if p.ONLY_CLASSIFY:
                total_loss = digit_loss
            else:
                total_loss = total_loss + p.LEARN_TO_CLASSIFY * digit_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        ######################################
        # Evaluate if it can distinguish constraint when using perfect visual samples
        # difference with training is model_visual_samplers are replaced with target_visual_samplers
        eval_target_visual = sample_visual_combination(target_symbolic_samplers, target_visual_samplers, p.COMBINATION_BATCH_SIZE)
        eval_model_visual = sample_visual_combination(model_symbolic_samplers, target_visual_samplers, p.COMBINATION_BATCH_SIZE)

        # Compute output
        eval_target_output = discriminator(eval_target_visual)
        eval_model_output = discriminator(eval_model_visual)

        # Compute loss
        eval_classifier_accuracy = 0.5 * ((eval_target_output>0.5).float().mean() + (eval_model_output<=0.5).float().mean())
        #eval_classifier_accuracy = 0.5 * ((eval_target_output>0.5).float().mean() + (eval_model_output<=0.5).float().mean())
        thresholds = torch.linspace(0, 1, 1000)
        eval_calibrated_accuracy = 0.5 * torch.max((eval_target_output > thresholds).float().mean(0)
                  + (eval_model_output <= thresholds).float().mean(0))

        ######################################
        # Log
        s.classifier_losses.append(classifier_loss.item())
        s.classifier_accuracies.append(classifier_accuracy.item())
        s.penalty_losses.append(penalty_loss.item())
        s.total_losses.append(total_loss.item())
        s.eval_classifier_accuracies.append(eval_classifier_accuracy.item())
        s.eval_calibrated_accuracies.append(eval_calibrated_accuracy.item())
        s.eval_target_outputs.append(eval_target_output.mean().item())
        s.eval_model_outputs.append(eval_model_output.mean().item())
        if iteration % 50 == 0:
            print '\nIteration', iteration
            #print 'Target', target_output.mean().item()
            #print 'Model', model_output.mean().item()
            print 'General divergence'
            print '\tClassifier Loss', summarize(s.classifier_losses)
            print '\tClassifier Accuracies', summarize(s.classifier_accuracies)
            print '\tPenalty Loss', summarize(s.penalty_losses)
            print 'Detect constraint'
            print '\tAccuracy with perfect model', summarize(s.eval_classifier_accuracies)
            print '\tAccuracy with perfect model', summarize(s.eval_calibrated_accuracies)
            print '\tEval Target', summarize(s.eval_target_outputs)
            print '\tEval Model', summarize(s.eval_model_outputs)
            print 'Individual digit classification'
            print '\tDigit losses', summarize(s.digit_losses)
            print '\tDigit accuracies', summarize(s.digit_accuracies)
            print 'Total Loss', summarize(s.total_losses)

            # Dump data
            with open('{}/stats.json'.format(run_dir), 'wb') as fp:
                json.dump(vars(s), fp, indent=4)


except KeyboardInterrupt:
    print 'interrupted'
