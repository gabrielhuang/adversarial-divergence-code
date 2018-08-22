import sys
sys.path.append('..')
import torch
from torchvision import datasets, transforms
from models import VAE, loss_function, Discriminator1, Discriminator2
from problems import get_problem
import numpy as np
#import matplotlib.pyplot as plt

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
    x = []
    for c in combination:
        sample = visual_samplers[c]()
        x.append(sample[None, :, :])
    visual_combination = torch.cat(x)[None, ...]
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

    old_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    new_gradient_penalty = torch.max((gradients**2).sum() - 1., torch.zeros(1))
    return new_gradient_penalty

##################


p = get_problem('sum_25', 'int', train_ratio=1.)


#####################################################
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
batch_size = 1

# Load all MNIST digits
test_digits = {}
test_loaders = {}
for i in xrange(10):
    print 'Loading digit', i
    test_digit = load_one_mnist_digit(i, train=False, debug=True)
    test_digits[i] = test_digit

    test_loader = torch.utils.data.DataLoader(test_digit, batch_size=batch_size, shuffle=True)
    test_loaders[i] = test_loader


#####################################################
def create_target_visual_sampler(i, test_loaders):
    def visual_sampler():
        return iter(test_loaders[int(i)]).next()[0][0, 0]

    return visual_sampler
target_visual_samplers = [create_target_visual_sampler(i, test_loaders) for i in xrange(10)]

# Build target
def create_model_visual_sampler(i, test_loaders):
    def visual_sampler():
        return vaes[i].generate(1)[0,0]

    return visual_sampler

class ModelVisualSampler(object):
    def __init__(self, vae, batch_size=32):
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
        return sample

#model_visual_samplers = [create_model_visual_sampler(i, test_loaders) for i in xrange(10)]
model_visual_samplers = [ModelVisualSampler(vaes[i]) for i in xrange(10)]


##########################################
# Pick target distribution
target_combinations = sum_25.train_positive

# Pick model joint distribution
#model_combinations = uniform.train_positive
model_combinations = sum_25.train_positive

##########################################
# Create discriminator
#discriminator = Discriminator1(with_sigmoid=True)
discriminator = Discriminator2()
optimizer = torch.optim.Adam(discriminator.parameters())

##########################################
criterion = torch.nn.BCELoss()

ITERATIONS = 1000
classifier_losses = []
penalty_losses = []
total_losses = []
PENALTY = 10.

def summarize(v):
    return '{:.3f} +/- {:.3f}'.format(np.mean(v), np.std(v)/np.sqrt(len(v)))

for iteration in xrange(ITERATIONS):
    ####################################
    #  Sample data

    # Sample combination from TARGET
    target_idx = np.random.choice(len(target_combinations))
    target_combination = target_combinations[target_idx]

    # Make visual
    target_visual = combination_to_visual(target_combination, target_visual_samplers)
    #plt.imshow(target_visual.resize(5 * 28, 28))

    # Sample combination from MODEL
    model_idx = np.random.choice(len(model_combinations))
    model_combination = model_combinations[target_idx]

    # Make visual by sampling from VAEs
    model_visual = combination_to_visual(target_combination, model_visual_samplers)
    #plt.imshow(model_visual.detach().resize(5 * 28, 28))

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

    # Compute penalty
    target_penalty = compute_gradient_penalty(discriminator, target_visual)
    model_penalty = compute_gradient_penalty(discriminator, model_visual)
    penalty_loss = PENALTY * (target_penalty + model_penalty)

    total_loss = classifier_loss + penalty_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    ######################################
    # Log
    classifier_losses.append(classifier_loss.item())
    penalty_losses.append(penalty_loss)
    total_losses.append(total_loss)
    if iteration % 50 == 0:
        print 'Iteration', iteration
        print 'Target', target_output.mean().item()
        print 'Model', model_output.mean().item()
        print 'Classifier Loss', summarize(classifier_losses.values())
        print 'Penalty Loss', summarize(penalty_losses.values())
        print 'Total Loss', summarize(total_losses.values())


