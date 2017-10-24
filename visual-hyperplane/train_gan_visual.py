import os
import time
from tqdm import tqdm
import argparse
import numpy as np
import torch
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import grad
import scipy.misc
from tensorboardX import SummaryWriter  # install with pip install git+https://github.com/lanpa/tensorboard-pytorch
from elastic_deform import elastic_deform, ElasticDeformCached

from models import Generator, Discriminator

resolutions = [32,64,128,256,512]

parser = argparse.ArgumentParser(description='Train GAN on MILA-8')

# general learning
parser.add_argument('--iterations', default=100000, type=int, help='iterations')
parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
parser.add_argument('--threads', default=8, type=int, help='number of threads for data loading')
parser.add_argument('--logdir', required=True, help='where to log samples and models')
parser.add_argument('--save-samples', default=100, type=int, help='save samples every')
parser.add_argument('--save-models', default=1000, type=int, help='save models every')
parser.add_argument('--log-every', default=100, type=int, help='log every N iterations')
parser.add_argument('--cuda', default=1, type=int, help='use cuda')
parser.add_argument('--validate-every', default=20, type=int, help='validate every N iterations')
parser.add_argument('--generate-samples', default=64, type=int, help='generate N samples')
parser.add_argument('--random-seed', default=1234, type=int, help='random seed')
parser.add_argument('--mnist', default='data', help='folder where MNIST is/will be downloaded')
parser.add_argument('--sample-rows', default=10, type=int, help='how many samples in tensorboard')

# task specific
parser.add_argument('--amount', default=25, type=int, help='target to sum up to')
parser.add_argument('--digits', default=5, type=int, help='how many digits per sequence')

# WGAN-GP specific
parser.add_argument('-p', '--penalty', default=10., type=float, help='gradient penalty')
parser.add_argument('--double-sided', default=0, type=int, help='use double sided penalty vs single sided')
parser.add_argument('--glr', default=1e-4, type=float, help='generator learning rate')
parser.add_argument('--dlr', default=1e-4, type=float, help='discriminator learning rate')
parser.add_argument('--batchnorm', default=1, type=int, help='whether to use batchnorm')
parser.add_argument('--latent', default=64, type=int, help='latent dimensions')
parser.add_argument('--critic-iterations', default=5, type=int, help='number of critic iterations')
arser.add_argument('--sigma', default=0.1, type=float, help='vae gaussian bandwidth')

args = parser.parse_args()
print args
assert args.resolution in resolutions
date = time.strftime('%Y-%m-%d.%H%M')
run_dir = '{}/gan-{}-{}-{}'.format(args.logdir, args.resolution, args.latent, date)
# Create models dir if deosnt exist
models_dir = '{}/models'.format(run_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

def get_gradient_penalty(netD, real_data, fake_data, double_sided, cuda):
    global gradients  # for debugging

    batch_size = real_data.size()[0]
    real_flat = real_data.view((batch_size, -1))
    fake_flat = fake_data.view((batch_size, -1))

    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_flat.size())  # broadcast over features
    if cuda:
        alpha = alpha.cuda()

    interpolates_flat = alpha*real_flat + (1-alpha)*fake_flat
    interpolates_flat = Variable(interpolates_flat, requires_grad=True)
    interpolates = interpolates_flat.view(*real_data.size())

    D_interpolates = netD(interpolates)

    ones = torch.ones(D_interpolates.size())
    zeros = torch.zeros(batch_size)
    if cuda:
        ones = ones.cuda()
        zeros = zeros.cuda()

    gradients = grad(outputs=D_interpolates, inputs=interpolates_flat,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    if double_sided:
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    else:
        gradient_penalty, __ = torch.max((gradients*gradients).sum(1)-1, 0)
        gradient_penalty = gradient_penalty.mean()

    return gradient_penalty


full, train, test = get_full_train_test(args.amount, range(10), args.digits, one_hot=False, validation=0.8, seed=args.random_seed)
train_visual = HyperplaneImageDataset(train, args.mnist, train=True)
test_visual = HyperplaneImageDataset(test, args.mnist, train=False)
train_loader = DataLoader(train_visual, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_visual, batch_size=args.batch_size, shuffle=True)
def infinite_data(loader):
    while True:
        for data in loader:
            yield data
train_iter = infinite_data(train_loader)
test_iter = infinite_data(test_loader)

# Prepare models
netD = Discriminator(args.latent, args.resolution, False)
netG = Generator(args.latent, args.resolution, args.batchnorm)
if args.cuda:
    netD = netD.cuda()
    netG = netG.cuda()
print 'Discriminator', netD
print 'Generator', netG

# Optimizer
optimizerD = Adam(netD.parameters(), lr=args.dlr, betas=(0.5, 0.9))
optimizerG = Adam(netG.parameters(), lr=args.glr, betas=(0.5, 0.9))

# Log
log = SummaryWriter(run_dir)
print 'Writing logs to {}'.format(run_dir)


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
        real_data, __ = train_iter.next()
        if args.cuda:
            real_data = real_data.cuda()
        real_data = Variable(real_data)
        if len(real_data) != args.batch_size:
            # the last batch might be smaller, skip it
            continue
        D_real = netD(real_data).mean()

        # Fake data
        # volatile: do not compute gradient for netG
        # stop gradient at fake_data
        fake_data = Variable(netG.generate(args.batch_size, volatile=True, use_cuda=args.cuda).data)
        D_fake = netD(fake_data).mean()

        # Costs
        gradient_penalty = args.penalty * get_gradient_penalty(
            netD, real_data.data, fake_data.data, double_sided=args.double_sided, cuda=args.cuda)
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
    fake_data = netG.generate(args.batch_size, volatile=False, use_cuda=args.cuda)
    D_fake = netD(fake_data).mean()

    # Costs
    G_cost = -D_fake

    # Train G but not D
    netG.zero_grad()
    G_cost.backward()
    optimizerG.step()

    # Write logs and save samples
    log.add_scalar('timePerIteration', time.time()-start_time, iteration)
    log.add_scalar('discriminatorCost', D_cost.cpu().data.numpy(), iteration)
    log.add_scalar('generatorCost', G_cost.cpu().data.numpy(), iteration)
    log.add_scalar('wasserstein', Wasserstein_D.cpu().data.numpy(), iteration)
    log.add_scalar('gradientPenalty', gradient_penalty.cpu().data.numpy(), iteration)

    # Export samples to tensorboard
    if iteration % args.save_samples == 0:
        gallery_train = torchvision.utils.make_grid(real_data.data, normalize=True, range=(0,1))
        gallery_gen = torchvision.utils.make_grid(fake_data.data, normalize=True, range=(0,1))
        log.add_image('train', gallery_train, iteration)
        log.add_image('generation', gallery_gen, iteration)
        print 'Saving samples to tensorboard'

    # Reconstructions
    if iteration % args.save_samples == 0:
        # Generate samples
        samples = vae.generate(args.sample_rows, use_cuda=args.cuda)

        # Process and log
        view_train = view_samples(data.data, args.sample_rows)
        view_gen = view_samples(samples.data, args.generate_samples)

        gallery_train = torchvision.utils.make_grid(view_train,
            nrow=args.digits, normalize=True, range=(0,1))
        gallery_gen = torchvision.utils.make_grid(view_gen,
            nrow=args.digits, normalize=True, range=(0,1))

        log.add_image('train', gallery_train, iteration)
        log.add_image('generation', gallery_gen, iteration)

    # Save models
    if iteration % args.save_models == 0:
        frame_str = '{:08d}'.format(iteration)
        print 'Saving models'
        torch.save(netD, '{}/discriminator_{}.torch'.format(models_dir, iteration))
        torch.save(netG, '{}/generator_{}.torch'.format(models_dir, iteration))
