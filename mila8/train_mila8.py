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
from elastic_deform import elastic_deform, ElasticDeformCached

import vae_mila8

resolutions = [32,64,128,256,512]

parser = argparse.ArgumentParser(description='Compute scattering transform of a given dataset.\nThe resulting tensor will have size (M_pad/2^J, M_pad/2^J, 1 + JL + L^2 J(J-1)/2')
parser.add_argument('--datadir', help='path to image folder')
parser.add_argument('--iterations', default=100000, type=int, help='iterations')
parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
parser.add_argument('--resolution', required=True, type=int, help='|'.join(map(str, resolutions)))
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
parser.add_argument('--deform', default=1, type=int, help='do random deformations')
parser.add_argument('--deform-alpha', default=8000., type=float, help='random deformation amplitude')
parser.add_argument('--deform-sigma', default=70, type=int, help='random deformation bandwidth')
parser.add_argument('--deform-cache', default=1000, type=int, help='random deformation cache')
parser.add_argument('--deform-reuse', default=32, type=int, help='how many times to reuse each deformation')
parser.add_argument('--validation', default=0.1, type=float, help='fraction of validation set')
parser.add_argument('--validate-every', default=20, type=int, help='validate every N iterations')
parser.add_argument('--generate-samples', default=64, type=int, help='generate N samples')

args = parser.parse_args()
print args
assert args.resolution in resolutions
date = time.strftime('%Y-%m-%d.%H%M')
run_dir = '{}/vae-{}-{}-{}'.format(args.logdir, args.resolution, args.latent, date)
# Create models dir if deosnt exist
models_dir = '{}/models'.format(run_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
args.deform_size = args.resolution
factor = args.deform_size / 256.
alpha = args.deform_alpha * factor**2
sigma = int(args.deform_sigma * factor)

if args.deform_cache and args.deform:
    elastic_deform_cached = ElasticDeformCached((args.deform_size, args.deform_size),
                                                alpha,
                                                sigma,
                                                args.deform_cache)

# Create deformation cache
def make_gray(data):
    data = data.mean(2)
    return data

def deform(data):
    if args.deform:
        if args.deform_cache:
            data = elastic_deform_cached.deform(data)
        else:
            data = elastic_deform(data, alpha, sigma, random_state=None)
    return data

def normalize(data, percentile=80):
    '''
    Push digit values close to 1, while background kept close to 0
    '''
    mask = data>0.
    if not np.any(mask):
        data_max = 1e-8
    else:
        data_max = np.percentile(data[mask], percentile)
    data = np.clip(data / data_max, 0, 1)
    return data

def resize(img):
    #img = scipy.misc.imresize(img, (args.resolution, args.resolution), interp='bicubic', mode='F')
    img = scipy.misc.imresize(img, (args.resolution, args.resolution), interp='bilinear', mode='F')
    return img

# Load MILA8 dataset
dataset = datasets.ImageFolder(args.datadir,
                        transform=transforms.Compose([
                            transforms.Scale(args.deform_size),
                            # PIL image
                            transforms.Lambda(lambda img: 1.-np.asarray(img)/255.),
                            # Numpy 256 x 256 x 3
                            transforms.Lambda(make_gray),
                            # Numpy 256 x 256
                            transforms.Lambda(deform),
                            # Numpy args.resolution x args.resolution
                            transforms.Lambda(normalize),
                            transforms.Lambda(lambda img: torch.Tensor(img[np.newaxis]))
                            # Torch 1 x args.resolution x args.resolution
                        ]))
dataset_len = len(dataset)
train_limit = dataset_len - int(args.validation*dataset_len)
train_loader = DataLoader(dataset, batch_size=args.batch_size,
                          num_workers=args.threads,
                          sampler=SubsetRandomSampler(range(0, train_limit)))
test_loader = DataLoader(dataset, batch_size=args.batch_size,
                          num_workers=args.threads,
                          sampler=SubsetRandomSampler(range(train_limit, dataset_len)))
def infinite_data(loader):
    while True:
        for data in loader:
            yield data
train_iter = infinite_data(train_loader)
test_iter = infinite_data(test_loader)

# Recompute cache every ... ?
if args.deform_reuse == 0:
    recompute_cache_every = (len(dataset) * args.deform_cache / args.batch_size)
else:
    recompute_cache_every = args.deform_reuse / args.batch_size
print 'Will recompute cache every {} iterations'.format(recompute_cache_every)

# Prepare models
vae = vae_mila8.VAE(args.latent, args.resolution, args.batchnorm)
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
        gallery_rec = torchvision.utils.make_grid(r_data.data, normalize=True, range=(0,1))
        gallery_train = torchvision.utils.make_grid(data.data, normalize=True, range=(0,1))
        log.add_image('train', gallery_train, iteration)
        log.add_image('reconstruction', gallery_rec, iteration)
        # Generate 100 images
        samples = vae.generate(args.generate_samples)
        gallery_gen = torchvision.utils.make_grid(samples.data, normalize=True, range=(0,1))
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

    # Recompute deformation cache
    if (args.deform
            and args.deform_cache
            and iteration % recompute_cache_every == 0):
        print '*'*32
        print 'Recomputing cache!'
        print '*'*32
        elastic_deform_cached.recompute()
