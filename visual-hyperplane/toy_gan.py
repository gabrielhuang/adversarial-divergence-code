import os, sys
sys.path.append(os.getcwd())

import random
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

MODE = 'wgan-gp' # wgan or wgan-gp
DATASET = 'swissroll' # 8gaussians, 25gaussians, swissroll
DIM = 256 # Model dimensionality
FIXED_GENERATOR = False # whether to hold the generator fixed at real data plus
                        # Gaussian noise, as in the plots in the paper
LAMBDA = .1 # Smaller lambda makes things faster for toy tasks, but isn't
            # necessary if you increase CRITIC_ITERS enough
CRITIC_ITERS = 5 # How many critic iterations per generator iteration
BATCH_SIZE = 256 # Batch size
ITERS = 100000 # how many generator iterations to train for
USE_CUDA = False


def get_gradient_penalty(netD, real_data, fake_data):
    global gradients  # for debugging

    batch_size = real_data.size()[0]
    real_flat = real_data.view((batch_size, -1))
    fake_flat = fake_data.view((batch_size, -1))

    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_flat.size())  # broadcast over features
    if USE_CUDA:
        alpha = alpha.cuda()

    interpolates_flat = alpha*real_flat + (1-alpha)*fake_flat
    interpolates_flat = Variable(interpolates_flat, requires_grad=True)
    interpolates = interpolates_flat.view(*real_data.size())

    D_interpolates = netD(interpolates)

    grad_outputs = torch.ones(D_interpolates.size())
    if USE_CUDA:
        grad_outputs = grad_outputs.cuda()

    gradients = grad(outputs=D_interpolates, inputs=interpolates_flat,
                     grad_outputs=grad_outputs,
                     create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


#########################
# Networks
########################

class Generator(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.dense1 = nn.Linear(2, DIM)
        self.dense2 = nn.Linear(DIM, DIM)
        self.dense3 = nn.Linear(DIM, DIM)
        self.dense4 = nn.Linear(DIM, 2)

    def forward(self, input):
        out = F.relu(self.dense1(input))
        out = F.relu(self.dense2(out))
        out = F.relu(self.dense3(out))
        out = self.dense4(out)
        return out

    def generate(self, batch_size, volatile=True):
        noise = torch.randn(batch_size, 2)
        if USE_CUDA:
            noise = noise.cuda()
        noise = Variable(noise, volatile=volatile)
        samples = self(noise)
        return samples


class Discriminator(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.dense1 = nn.Linear(2, DIM)
        self.dense2 = nn.Linear(DIM, DIM)
        self.dense3 = nn.Linear(DIM, DIM)
        self.dense4 = nn.Linear(DIM, 1)

    def forward(self, input):
        out = F.relu(self.dense1(input))
        out = F.relu(self.dense2(out))
        out = F.relu(self.dense3(out))
        out = self.dense4(out)
        return out



################################
# MAIN
###############################

######################
# Create Models
######################
netG = Generator()
netD = Discriminator()
print netG
print netD
if USE_CUDA:
    netD = netD.cuda()
    netG = netG.cuda()

optimizerD = Adam(netD.parameters(), lr=5e-5, betas=(0.5, 0.9))
optimizerG = Adam(netG.parameters(), lr=5e-5, betas=(0.5, 0.9))

######################
# Load data
######################
def inf_train_gen():
    if DATASET == '25gaussians':

        dataset = []
        for i in xrange(100000 / 25):
            for x in xrange(-2, 3):
                for y in xrange(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    dataset.append(point)
        dataset = np.array(dataset, dtype='float32')
        np.random.shuffle(dataset)
        dataset /= 2.828  # stdev
        while True:
            for i in xrange(len(dataset) / BATCH_SIZE):
                yield dataset[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

    elif DATASET == 'swissroll':

        while True:
            data = sklearn.datasets.make_swiss_roll(
                n_samples=BATCH_SIZE,
                noise=0.25
            )[0]
            data = np.ascontiguousarray(data.astype('float32')[:, [0, 2]])
            data /= 7.5  # stdev plus a little
            yield data

    elif DATASET == '8gaussians':

        scale = 2.
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        while True:
            dataset = []
            for i in xrange(BATCH_SIZE):
                point = np.random.randn(2) * .02
                center = random.choice(centers)
                point[0] += center[0]
                point[1] += center[1]
                dataset.append(point)
            dataset = np.array(dataset, dtype='float32')
            dataset /= 1.414  # stdev
            yield dataset

###########################

###########################
frame_index = [0]
if not os.path.exists('plots'):
    os.makedirs('plots')
def generate_image(true_dist, samples):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    plt.clf()

    plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', marker='+')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    plt.savefig('plots/frame' + str(frame_index[0]) + '.jpg')
    frame_index[0] += 1


######################
# Main loop
######################
gen = inf_train_gen()
for iteration in tqdm(xrange(ITERS)):
    start_time = time.time()

    ############################
    # (1) Update D network
    ###########################

    # Require gradient for netD
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    for iter_d in xrange(CRITIC_ITERS):
        # Real data
        real_data = torch.Tensor(gen.next())
        if USE_CUDA:
            real_data = real_data.cuda()
        real_data = Variable(real_data)
        D_real = netD(real_data).mean()

        # Fake data
        # volatile: do not compute gradient for netG
        # stop gradient at fake_data
        fake_data = Variable(netG.generate(BATCH_SIZE, volatile=True).data)
        D_fake = netD(fake_data).mean()

        # Costs
        gradient_penalty = LAMBDA * get_gradient_penalty(
            netD, real_data.data, fake_data.data)
        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake

        # Train D but not G
        netD.zero_grad()
        D_cost.backward()
        optimizerD.step()

    ############################
    # (2) Update G network
    ###########################

    # Do not compute gradient for netD
    for p in netD.parameters():
        p.requires_grad = False  # to avoid computation

    # Generate fake data
    # volatile: compute gradients for netG
    fake_data = netG.generate(BATCH_SIZE, volatile=False)
    D_fake = netD(fake_data).mean()

    # Costs
    G_cost = -D_fake

    # Train G but not D
    netG.zero_grad()
    G_cost.backward()
    optimizerG.step()

    # Write logs and save samples
    if iteration % 100 == 99:
        generate_image(real_data.data.numpy(), fake_data.data.numpy())


