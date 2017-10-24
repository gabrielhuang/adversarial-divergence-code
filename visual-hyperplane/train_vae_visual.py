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
import scipy.misc
from tensorboardX import SummaryWriter  # install with pip install git+https://github.com/lanpa/tensorboard-pytorch
from hyperplane_dataset import HyperplaneImageDataset, get_full_train_test

from models import VAE, UnconstrainedVAE


parser = argparse.ArgumentParser(description='Train VAE on MILA-8')
parser.add_argument('--amount', default=25, type=int, help='target to sum up to')
parser.add_argument('--digits', default=5, type=int, help='how many digits per sequence')
parser.add_argument('--iterations', default=100000, type=int, help='iterations')
parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
parser.add_argument('--batchnorm', default=1, type=int, help='whether to use batchnorm')
parser.add_argument('--latent', default=64, type=int, help='latent dimensions')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--threads', default=8, type=int, help='number of threads for data loading')
parser.add_argument('--logdir', required=True, help='where to log samples and models')
parser.add_argument('--save-samples', default=100, type=int, help='save samples every')
parser.add_argument('--save-models', default=1000, type=int, help='save models every')
parser.add_argument('--log-every', default=100, type=int, help='log every N iterations')
parser.add_argument('--sigma', default=0.1, type=float, help='vae gaussian bandwidth')
parser.add_argument('--cuda', default=1, type=int, help='use cuda')
parser.add_argument('--validation', default=0.1, type=float, help='fraction of validation set')
parser.add_argument('--validate-every', default=20, type=int, help='validate every N iterations')
parser.add_argument('--generate-samples', default=64, type=int, help='generate N samples')
parser.add_argument('--random-seed', default=1234, type=int, help='random seed')
parser.add_argument('--mnist', default='data', help='folder where MNIST is/will be downloaded')
parser.add_argument('--sample-rows', default=10, type=int, help='how many samples in tensorboard')


args = parser.parse_args()
print args
date = time.strftime('%Y-%m-%d.%H%M')
run_dir = '{}/vae-{}-{}'.format(args.logdir, args.latent, date)
# Create models dir if deosnt exist
models_dir = '{}/models'.format(run_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

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
vae = UnconstrainedVAE(args.latent, args.digits, args.batchnorm)
if args.cuda:
    vae = vae.cuda()
print vae

# Optimizer
optimizer = Adam(vae.parameters(), lr=args.lr, betas=(0.5, 0.9))

# Log
log = SummaryWriter(run_dir)
print 'Writing logs to {}'.format(run_dir)

def get_loss(data, r_data, mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size()[0]
    MSE = 0.5 / args.sigma * torch.sum((r_data - data)**2) / data.size()[0]
    loss = KLD + MSE
    return KLD, MSE, loss

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
    data, label = train_iter.next()
    if args.cuda:
        data = data.cuda()
    data = Variable(data)

    # Compute
    r_data, mu, logvar = vae(data)

    # Loss
    KLD, MSE, loss = get_loss(data, r_data, mu, logvar)

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
            nrow=args.digits, normalize=True, range=(0,1))
        gallery_train = torchvision.utils.make_grid(view_train,
            nrow=args.digits, normalize=True, range=(0,1))
        log.add_image('train', gallery_train, iteration)
        log.add_image('reconstruction', gallery_rec, iteration)

        # Generate 100 images
        samples = vae.generate(args.sample_rows, use_cuda=args.cuda)
        view_gen = view_samples(samples.data, args.generate_samples)
        gallery_gen = torchvision.utils.make_grid(view_gen,
            nrow=args.digits, normalize=True, range=(0,1))
        log.add_image('generation', gallery_gen, iteration)

    # Models
    if iteration % args.save_models == 0:
        fname = '{}/vae-{}.torch'.format(models_dir, iteration)
        print 'Saving models', fname
        torch.save(vae, fname)

    # Evaluate
    if iteration % args.validate_every == 0:
        data, label = test_iter.next()
        if args.cuda:
            data = data.cuda()
        data = Variable(data)
        r_data, mu, logvar = vae(data)
        KLD, MSE, loss = get_loss(data, r_data, mu, logvar)
        log.add_scalar('loss_VAL', loss.data[0], iteration)
        log.add_scalar('KL_VAL', KLD.data[0], iteration)
        log.add_scalar('MSE_VAL', MSE.data[0], iteration)

