######################
# ARGUMENTS
######################
import os
import argparse
import time


parser = argparse.ArgumentParser(description='Train a GAN on digits')
parser.add_argument('-l', '--latent', default=100, type=int, help='number of latent dimensions')
parser.add_argument('-p', '--penalty', default=10., type=float, help='gradient penalty')
parser.add_argument('-i', '--iterations', default=200000, type=int, help='number of iterations')
parser.add_argument('--batch-size', default=64, type=int, help='minibatch batch size')
parser.add_argument('--critic-iterations', default=5, type=int, help='number of critic iterations')
parser.add_argument('--model', default='gan')
parser.add_argument('--logdir', required=True, help='where to log samples and models')
parser.add_argument('--double-sided', default=True, action='store_true', help='whether to use double-sided penalty')
parser.add_argument('--save-samples', default=500, type=int, help='save samples every N iterations')
parser.add_argument('--save-models', default=5000, type=int, help='save models every N iterations')
parser.add_argument('--glr', default=1e-4, type=float, help='generator learning rate')
parser.add_argument('--dlr', default=1e-4, type=float, help='discriminator learning rate')
parser.add_argument('--use-cuda', default=False, type=bool, help='whether to use cuda')

args = parser.parse_args()
date = time.strftime('%Y-%m-%d.%H%M')
run_dir = '{}/{}-{}'.format(args.logdir, args.model, date)
# Create args.logdir and run_dir if doesnt exist (implicitly below)
# Create samples dir if deosnt exist
samples_dir = '{}/samples'.format(run_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)
# Create models dir if deosnt exist
models_dir = '{}/models'.format(run_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


######################
# Imports
######################
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.autograd import Variable, grad
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from hyperplane_dataset import  HyperplaneCachedDataset


######################
# Constant
######################
AMOUNT = 25
NB_DIGITS = 5
DIM = 10


######################
# Utilities
######################
if args.use_cuda:
    ZEROS = Variable(torch.zeros((args.batch_size)).cuda())
else:
    ZEROS = Variable(torch.zeros((args.batch_size)))


def get_gradient_penalty(netD, real_data, fake_data, double_sided=True):
    global gradients  # for debugging

    batch_size = real_data.size()[0]
    real_flat = real_data.view((batch_size, -1))
    fake_flat = fake_data.view((batch_size, -1))

    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_flat.size())  # broadcast over features
    if args.use_cuda:
        alpha = alpha.cuda()

    interpolates_flat = alpha*real_flat + (1-alpha)*fake_flat
    interpolates_flat = Variable(interpolates_flat, requires_grad=True)
    interpolates = interpolates_flat.view(*real_data.size())

    D_interpolates = netD(interpolates)

    grad_outputs = torch.ones(D_interpolates.size())
    if args.use_cuda:
        grad_outputs = grad_outputs.cuda()

    gradients = grad(outputs=D_interpolates, inputs=interpolates_flat,
                     grad_outputs=grad_outputs,
                     create_graph=True, retain_graph=True, only_inputs=True)[0]

    if double_sided:
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    else:
        gradient_penalty = torch.max((gradients*gradients).sum(1)-1, ZEROS).mean()

    return gradient_penalty


#########################
# Networks
########################

class DigitGenerator(nn.Module):

    def __init__(self, latent):
        nn.Module.__init__(self)
        self.latent = latent
        N = NB_DIGITS * DIM
        self.dense1 = nn.Linear(latent, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, N)

    def forward(self, input):
        out = F.relu(self.dense1(input))
        out = F.relu(self.dense2(out))
        out = self.dense3(out)
        out = F.softmax(out)
        return out.view(-1, NB_DIGITS, DIM)

    def generate(self, batch_size, volatile=True):
        noise = torch.randn(batch_size, self.latent)
        if args.use_cuda:
            noise = noise.cuda()
        noise = Variable(noise, volatile=volatile)
        samples = self(noise)
        return samples


class DigitDiscriminator(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)

        N = NB_DIGITS * DIM
        self.dense1 = nn.Linear(N, 128)
        self.dense2 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 1)

    def forward(self, input):
        N = NB_DIGITS * DIM
        out = input.view(-1, N)
        out = F.relu(self.dense1(out))
        out = F.relu(self.dense2(out))
        out = self.dense3(out)
        return out.view(-1, 1)


################################
# MAIN
###############################

######################
# Create Models
######################
netG = DigitGenerator(args.latent)
netD = DigitDiscriminator()
print netG
print netD
if args.use_cuda:
    netD = netD.cuda()
    netG = netG.cuda()
optimizerD = Adam(netD.parameters(), lr=args.dlr, betas=(0.5, 0.9))
optimizerG = Adam(netG.parameters(), lr=args.glr, betas=(0.5, 0.9))

######################
# Load data
######################
dataset = HyperplaneCachedDataset(AMOUNT, range(DIM), NB_DIGITS, one_hot=True)
data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


def infinite_data(loader):
    while True:
        for data in loader:
            yield data


data_iter = infinite_data(data_loader)


######################
# Loggers
######################
log = SummaryWriter(run_dir)
print 'Writing logs to {}'.format(run_dir)
# Generate subdirectory


# Write all parameters to tensorboard
log.add_text('archD', str(netD), 0)
log.add_text('archG', str(netG), 0)
log.add_text('args', str(args), 0)


######################
# Main loop
######################
for iteration in tqdm(xrange(args.iterations)):
    start_time = time.time()

    ############################
    # (1) Update D network
    ###########################

    # Require gradient for netD
    for p in netD.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update

    for iter_d in xrange(args.critic_iterations):
        # Real data
        real_data = torch.Tensor(data_iter.next())
        if args.use_cuda:
            real_data = real_data.cuda()
        real_data = Variable(real_data)
        if len(real_data) != args.batch_size:
            # the last batch might be smaller, skip it
            continue
        D_real = netD(real_data).mean()

        # Fake data
        # volatile: do not compute gradient for netG
        # stop gradient at fake_data
        fake_data = Variable(netG.generate(args.batch_size, volatile=True).data)
        D_fake = netD(fake_data).mean()

        # Costs
        gradient_penalty = args.penalty * get_gradient_penalty(
            netD, real_data.data, fake_data.data, double_sided=args.double_sided)
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
    fake_data = netG.generate(args.batch_size, volatile=False)
    D_fake = netD(fake_data).mean()

    # Costs
    G_cost = -D_fake

    # Train G but not D
    netG.zero_grad()
    G_cost.backward()
    optimizerG.step()

    # Write logs and save samples
    log.add_scalar('timePerIteration', time.time() - start_time, iteration)
    log.add_scalar('discriminatorCost', D_cost.cpu().data.numpy(), iteration)
    log.add_scalar('generatorCost', G_cost.cpu().data.numpy(), iteration)
    log.add_scalar('wasserstein', Wasserstein_D.cpu().data.numpy(), iteration)
    log.add_scalar('gradientPenalty', gradient_penalty.cpu().data.numpy(), iteration)

    # Calculate dev loss and generate samples every 100 iters
    if iteration % args.save_samples == 0:
        filename = '{}/softmax_{}.npy'.format(samples_dir, iteration)
        samples = netG.generate(100)  # by default generate 100 samples
        samples = samples.data.cpu().numpy()

        np.save(filename, samples)
        filename = '{}/digits_{}.npy'.format(samples_dir, iteration)
        digits = np.argmax(samples, axis=-1)
        np.save(filename, digits)
        print 'Saving samples to {}'.format(filename)

    if iteration % args.save_models == 0:
        frame_str = '{:08d}'.format(iteration)
        print 'Saving models'
        torch.save(netD, '{}/discriminator_{}.torch'.format(models_dir, iteration))
        torch.save(netG, '{}/generator_{}.torch'.format(models_dir, iteration))






