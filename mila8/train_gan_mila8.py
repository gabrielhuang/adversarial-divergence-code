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
parser.add_argument('--datadir', help='path to image folder')
parser.add_argument('--iterations', default=100000, type=int, help='iterations')
parser.add_argument('-p', '--penalty', default=10., type=float, help='gradient penalty')
parser.add_argument('--double-sided', default=0, type=int, help='use double sided penalty vs single sided')
parser.add_argument('--batch-size', default=64, type=int, help='minibatch size')
parser.add_argument('--resolution', required=True, type=int, help='|'.join(map(str, resolutions)))
parser.add_argument('--glr', default=1e-4, type=float, help='generator learning rate')
parser.add_argument('--dlr', default=1e-4, type=float, help='discriminator learning rate')
parser.add_argument('--batchnorm', default=1, type=int, help='whether to use batchnorm')
parser.add_argument('--latent', default=64, type=int, help='latent dimensions')
parser.add_argument('--critic-iterations', default=5, type=int, help='number of critic iterations')
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
parser.add_argument('--generate-samples', default=64, type=int, help='generate N samples')
parser.add_argument('--validation', default=0.1, type=float, help='fraction of validation set')

args = parser.parse_args()
print args
assert args.resolution in resolutions
date = time.strftime('%Y-%m-%d.%H%M')
run_dir = '{}/gan-{}-{}-{}'.format(args.logdir, args.resolution, args.latent, date)
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
    recompute_cache_every = args.deform_cache * args.deform_reuse / args.batch_size
recompute_cache_every = max(1, recompute_cache_every)
print 'Will recompute cache every {} iterations'.format(recompute_cache_every)

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

    # Save models
    if iteration % args.save_models == 0:
        frame_str = '{:08d}'.format(iteration)
        print 'Saving models'
        torch.save(netD, '{}/discriminator_{}.torch'.format(models_dir, iteration))
        torch.save(netG, '{}/generator_{}.torch'.format(models_dir, iteration))
