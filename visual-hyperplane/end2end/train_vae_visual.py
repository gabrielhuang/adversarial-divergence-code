import os
import time
from tqdm import tqdm
import argparse
import cPickle as pickle
import numpy as np
import json
import torch
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import functional as F
import scipy.misc
from tensorboardX import SummaryWriter  # install with pip install git+https://github.com/lanpa/tensorboard-pytorch

import sys
sys.path.append('..')
from common.vae_models import VAE, UnconstrainedVAE
from common import digits_sampler


parser = argparse.ArgumentParser(description='Train VAE on MILA-8')

# general learning
parser.add_argument('--iterations', default=100000, type=int, help='iterations')
parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
parser.add_argument('--threads', default=8, type=int, help='number of threads for data loading')
parser.add_argument('--logdir', default='log_visual', help='where to log samples and models')
parser.add_argument('--save-samples', default=100, type=int, help='save samples every')
parser.add_argument('--save-models', default=1000, type=int, help='save models every')
parser.add_argument('--log-every', default=100, type=int, help='log every N iterations')
parser.add_argument('--cuda', default=1, type=int, help='use cuda')
parser.add_argument('--validate-every', default=20, type=int, help='validate every N iterations')
parser.add_argument('--generate-samples', default=64, type=int, help='generate N samples')
parser.add_argument('--mnist', default='data', help='folder where MNIST is/will be downloaded')
parser.add_argument('--sample-rows', default=10, type=int, help='how many samples in tensorboard')

# task specific
parser.add_argument('--data', default='sum_25.pkl', help='pickled dataset')

# VAE specific
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--latent-global', default=200, type=int, help='latent dimensions')
parser.add_argument('--sigma', default=0.1, type=float, help='vae gaussian bandwidth')
parser.add_argument('--model', default='constrained', choices=['unconstrained', 'constrained'], help="The vae model to choose from ('unconstrained','constrained')")

args = parser.parse_args()
print args
date = time.strftime('%Y-%m-%d.%H%M')
run_dir = '{}/vae-{}-{}'.format(args.logdir, args.latent_global, date)
# Create models dir if deosnt exist
models_dir = '{}/models'.format(run_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
# Create samples dir if deosnt exist
samples_dir = '{}/samples'.format(run_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)

device = 'cuda:0' if args.cuda else 'cpu'
#####################################################
# Load data for side task
# Load individual MNIST digits
test_visual_samplers = []
all_mnist_digit = digits_sampler.load_all_mnist_digits(train=False)
for i in xrange(10):
    print 'Loading digit', i
    digit_test_iter = digits_sampler.make_infinite(
        torch.utils.data.DataLoader(all_mnist_digit[i], batch_size=1, shuffle=True))
    test_visual_samplers.append(digits_sampler.DatasetVisualSampler(digit_test_iter))
print 'Not using any transforms, ensure models match that assumption!'
#####################################################
print 'Loading problem {}'.format(args.data)
with open(args.data, 'rb') as fp:
    problem = pickle.load(fp)
print problem
#####################################################

# Dump parameters
with open('{}/args.json'.format(run_dir), 'wb') as f:
    json.dump(vars(args), f, indent=4)

# Prepare models
if args.model == 'unconstrained':
    vae = UnconstrainedVAE(args.latent_global, 5, batchnorm=True)
elif args.model == 'constrained':
    vae = VAE(5, args.latent_global, batchnorm=True)
else:
    raise ValueError('Model not specified.')
if args.cuda:
    vae = vae.cuda()
print vae

# Optimizer
optimizer = Adam(vae.parameters(), lr=args.lr, betas=(0.5, 0.9))

# Log
log = SummaryWriter(run_dir)
print 'Writing logs to {}'.format(run_dir)

def get_loss_mse(data, r_data, mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size()[0]
    MSE = 0.5 / args.sigma * torch.sum((r_data - data)**2) / data.size()[0]
    loss = KLD + MSE
    return KLD, MSE, loss


def get_loss_bce(data, r_data, mu, logvar):
    BCE = F.binary_cross_entropy(r_data.view(len(r_data), -1), data.view(len(data), -1), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR,
    # 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    loss = KLD + BCE

    return KLD, BCE, loss


def view_samples(data, max_samples=None):
    '''
    reshape from (batch, args.digits, 28, 28) to (batch*args.digits, 1, 28, 28)
    '''
    if max_samples:
        data = data[:min(max_samples, len(data))]
    size = data.size()
    return data.view(-1, 1, size[-2], size[-1])

# Actual training
losses = []
for iteration in tqdm(xrange(args.iterations)):

    # Sample visual combination
    data = digits_sampler.sample_visual_combination(
        problem.train_positive,
        test_visual_samplers,
        args.batch_size).to(device)

    # Compute
    r_data, mu, logvar = vae(data)

    # Loss
    KLD, MSE, loss = get_loss_bce(data, r_data, mu, logvar)

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log
    losses.append(loss.data[0])
    if iteration % args.log_every == 0:
        print 'Loss', losses[-1]
        print 'KL', KLD.data[0]
        print 'MSE', MSE.data[0]
    log.add_scalar('loss_TRAIN', loss.data[0], iteration)
    log.add_scalar('KL_TRAIN', KLD.data[0], iteration)
    log.add_scalar('MSE_TRAIN', MSE.data[0], iteration)

    # Reconstructions
    if iteration % args.save_samples == 0:
        # Reconstruct
        view_rec = view_samples(r_data.data, args.sample_rows)
        view_train = view_samples(data.data, args.sample_rows)

        gallery_rec = torchvision.utils.make_grid(view_rec,
            nrow=5, normalize=True, range=(0,1))
        gallery_train = torchvision.utils.make_grid(view_train,
            nrow=5, normalize=True, range=(0,1))
        log.add_image('train', gallery_train, iteration)
        log.add_image('reconstruction', gallery_rec, iteration)

        # Generate 100 images
        samples = vae.generate(args.sample_rows, use_cuda=args.cuda)
        view_gen = view_samples(samples.data, args.generate_samples)
        gallery_gen = torchvision.utils.make_grid(view_gen,
            nrow=5, normalize=True, range=(0,1))
        log.add_image('generation', gallery_gen, iteration)

        iteration_fixed = '{:08d}'.format(iteration)
        torchvision.utils.save_image(gallery_train, '{}/train_{}.png'.format(samples_dir, iteration_fixed))
        torchvision.utils.save_image(gallery_gen, '{}/generated_{}.png'.format(samples_dir, iteration_fixed))
        torchvision.utils.save_image(gallery_rec, '{}/reconstruction_{}.png'.format(samples_dir, iteration_fixed))

    # Models
    if iteration % args.save_models == 0:
        fname = '{}/vae-{}.torch'.format(models_dir, iteration)
        print 'Saving models', fname
        torch.save(vae, fname)

    # Evaluate
    if iteration % args.validate_every == 0:
        # Sample visual combination
        data = digits_sampler.sample_visual_combination(
            problem.test_positive,
            test_visual_samplers,
            args.batch_size).to(device)

        data = Variable(data)
        r_data, mu, logvar = vae(data)
        KLD, MSE, loss = get_loss_bce(data, r_data, mu, logvar)
        log.add_scalar('loss_VAL', loss.data[0], iteration)
        log.add_scalar('KL_VAL', KLD.data[0], iteration)
        log.add_scalar('MSE_VAL', MSE.data[0], iteration)
