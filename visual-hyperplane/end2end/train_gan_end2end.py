import os
import time
from tqdm import tqdm
import argparse
import json
import cPickle as pickle
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import grad
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter  # install with pip install git+https://github.com/lanpa/tensorboard-pytorch

import sys
sys.path.append('..')
from common.models_end2end import (
    ConstrainedImageDiscriminator,
    ConstrainedImageGenerator,
    UnconstrainedImageDiscriminator,
    UnconstrainedImageGenerator)
from common.models import GeneratorCNN, DiscriminatorCNN
from common.gradient_penalty import compute_gradient_penalty_logits
from common import digits_sampler


#########################################
parser = argparse.ArgumentParser(description='Train GAN with visual hyperplane')

# general learning
parser.add_argument('--iterations', default=100000, type=int, help='iterations')
parser.add_argument('--digit-batch-size', default=64, type=int, help='minibatch size')
parser.add_argument('--combination-batch-size', default=12, type=int, help='combination minibatch size')
parser.add_argument('--threads', default=8, type=int, help='number of threads for data loading')
parser.add_argument('--logdir', default='gan_output', help='where to log samples and models')
parser.add_argument('--save-samples', default=100, type=int, help='save samples every')
parser.add_argument('--save-models', default=1000, type=int, help='save models every')
parser.add_argument('--log-every', default=100, type=int, help='log every N iterations')
parser.add_argument('--cuda', default=1, type=int, help='use cuda')
parser.add_argument('--validate-every', default=20, type=int, help='validate every N iterations')
parser.add_argument('--generate-samples', default=64, type=int, help='generate N samples')
parser.add_argument('--mnist', default='../mnist', help='folder where MNIST is/will be downloaded')
parser.add_argument('--sample-rows', default=10, type=int, help='how many samples in tensorboard')

# task specific
parser.add_argument('--data', default='sum_25.pkl', help='pickled dataset')

# GAN specific
parser.add_argument('--gp', default=10., type=float, help='gradient penalty')
parser.add_argument('--glr', default=1e-4, type=float, help='generator learning rate')
parser.add_argument('--dlr', default=1e-4, type=float, help='discriminator learning rate')
##### THIS PARAMETER IS IGNORED NOW
##### NO CONSISTENCY CHECK WITH GENERATOR@
parser.add_argument('--latent-local', default=100, type=int, help='local latent dimensions for each image')
parser.add_argument('--latent-global', default=200, type=int, help='global latent dimensions')
#parser.add_argument('--model-generator', default='constrained', choices=['constrained','unconstrained'], help='c|u')
#parser.add_argument('--model-discriminator', default='constrained', choices=['constrained','unconstrained'], help='constrained|unconstrained|semi')
parser.add_argument('--generator', default='ConstrainedImageGenerator', help='architecture')
parser.add_argument('--discriminator', default='ConstrainedImageDiscriminator', help='architecture')
#parser.add_argument('--filters', default=32, type=int, help='size of disc/gen in unconstrained case')

parser.add_argument('--side-task', default=1, help='Use side task')


args = parser.parse_args()
short = {'constrained': 'C', 'unconstrained': 'U', 'semi': 'S'}
print args

device = 'cuda:0' if args.cuda else 'cpu'
#######################################
# Create all folders

date = time.strftime('%Y-%m-%d.%H%M')
run_dir = '{}/{}'.format(args.logdir, date)
#run_dir = '{}/gan-d{}g{}-{}'.format(args.logdir, short[args.model_discriminator],
#                                short[args.model_generator], date)
# Create models dir if does not exist
models_dir = '{}/models'.format(run_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
# Create models dir if does not exist
samples_dir = '{}/samples'.format(run_dir)
if not os.path.exists(samples_dir):
    os.makedirs(samples_dir)
########################################
print 'Dumping arguments'
with open('{}/args.json'.format(run_dir),'wb') as fp:
    json.dump(vars(args), fp, indent=4)
########################################
def reshape_combinations(data, max_samples=None):
    '''
    reshape from (batch, args.digits, 28, 28) to (batch*args.digits, 1, 28, 28)
    '''
    if max_samples:
        data = data[:min(max_samples, len(data))]
    size = data.size()
    return data.view(-1, 1, size[-2], size[-1])
#####################################################
log = SummaryWriter(run_dir)
print 'Writing logs to {}'.format(run_dir)
#####################################################
print 'Loading problem {}'.format(args.data)
with open(args.data, 'rb') as fp:
    problem = pickle.load(fp)
print problem
#####################################################
print 'Loading MNIST digits'

transform = transforms.Compose([
    #transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load individual MNIST digits
test_visual_samplers = []
all_mnist_digit = digits_sampler.load_all_mnist_digits(train=False)
for i in xrange(10):
    print 'Loading digit', i
    digit_test_iter = digits_sampler.make_infinite(
        torch.utils.data.DataLoader(all_mnist_digit[i], batch_size=1, shuffle=True))
    test_visual_samplers.append(digits_sampler.DatasetVisualSampler(digit_test_iter))
#####################################################
print 'Building models'

# Prepare models
#if args.model_discriminator == 'unconstrained':
#    netD = UnconstrainedImageDiscriminator(5, args.filters)
#elif args.model_discriminator == 'constrained':
#    netD = ConstrainedImageDiscriminator(5)
#if args.model_generator == 'unconstrained':
#    netG = UnconstrainedImageGenerator(args.latent_global, 5, args.filters)
#else:
#    netG = ConstrainedImageGenerator(args.latent_global, args.latent_local, 5)
DiscriminatorClass = eval(args.discriminator)
GeneratorClass = eval(args.generator)

netD = DiscriminatorClass()
netG = GeneratorClass()


netD.to(device)
netG.to(device)
print 'Discriminator', netD
print 'Generator', netG
#####################################################
# Optimizer
optimizerD = Adam(netD.parameters(), lr=args.dlr, betas=(0.5, 0.9))
optimizerG = Adam(netG.parameters(), lr=args.glr, betas=(0.5, 0.9))
#####################################################
fixed_noise = torch.randn(args.combination_batch_size, args.latent_global, device=device)
criterion = nn.BCELoss()
real_label = 1
fake_label = 0

######################
# Main loop
######################
for iteration in tqdm(xrange(args.iterations)):
    start_time = time.time()

    ############################
    # Side-task
    ############################
    netD.zero_grad()

    if args.side_task:
        # Sample visual combination
        p_digit_images = digits_sampler.sample_visual_combination(
            problem.train_positive,
            test_visual_samplers,
            args.combination_batch_size,
            transform).to(device)
        q_digit_images = digits_sampler.sample_visual_combination(
            problem.train_negative,
            test_visual_samplers,
            args.combination_batch_size,
            transform).to(device)

        # Compute output
        p_out = netD(p_digit_images)
        q_out = netD(q_digit_images)

        p_target = torch.ones(len(p_out)).to(device)  # REAL is ONE, FAKE is ZERO
        q_target = torch.zeros(len(q_out)).to(device)  # REAL is ONE, FAKE is ZERO

        # Compute loss
        p_loss = criterion(p_out, p_target)
        q_loss = criterion(q_out, q_target)
        classifier_loss = 0.5 * (p_loss + q_loss)
        classifier_accuracy = 0.5 * ((p_out > 0.5).float().mean() + (q_out <= 0.5).float().mean())

        # No penalty right now
        side_loss = classifier_loss# + penalty_loss
        side_loss.backward()

        log.add_scalar('sideLoss', side_loss.item(), iteration)


    ####################################
    # Sample visual combination
    real = digits_sampler.sample_visual_combination(
        problem.train_positive,
        test_visual_samplers,
        args.combination_batch_size,
        transform).to(device)


    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    # train with real
    #netD.zero_grad()
    batch_size = real.size(0)
    label = torch.full((batch_size,), real_label, device=device)

    output = netD(real)
    errD_real = criterion(output, label)
    errD_real.backward()
    D_x = output.mean().item()

    # train with fake
    noise = torch.randn(batch_size, args.latent_global, device=device)
    fake = netG(noise)
    label.fill_(fake_label)
    output = netD(fake.detach())
    errD_fake = criterion(output, label)
    errD_fake.backward()
    D_G_z1 = output.mean().item()

    # gradient penalty
    gp = 0.
    if args.gp > 0.:
        gp_real = args.gp * compute_gradient_penalty_logits(netD, real)
        gp_fake = args.gp * compute_gradient_penalty_logits(netD, fake.detach())
        gradient_penalty = gp_real + gp_fake
        gradient_penalty.backward()
        gp = gradient_penalty.item()

    errD = errD_real + errD_fake
    optimizerD.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    netG.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
    output = netD(fake)
    errG = criterion(output, label)
    errG.backward()
    D_G_z2 = output.mean().item()
    optimizerG.step()



    ############################
    # Evaluation
    ############################

    ############################
    # Logs
    ###########################

    # Write logs and save samples
    log.add_scalar('timePerIteration', time.time()-start_time, iteration)
    log.add_scalar('lossG', errG.item(), iteration)
    log.add_scalar('lossD', errD.item(), iteration)
    log.add_scalar('D_real', D_x, iteration)
    log.add_scalar('D_fake1', D_G_z1, iteration)
    log.add_scalar('D_fake2', D_G_z2, iteration)
    log.add_scalar('gradientPenalty', gp, iteration)

    # Reconstructions
    if iteration % args.save_samples == 0:
        # Generate samples
        random_noise = torch.randn(batch_size, args.latent_global, device=device)
        random_samples = netG(noise)
        fixed_samples = netG(fixed_noise)

        # Process and log
        random_img = reshape_combinations(random_samples.detach(), args.sample_rows)
        fixed_img = reshape_combinations(fixed_samples.detach(), args.sample_rows)
        reference_img = reshape_combinations(real.detach(), args.sample_rows)

        random_gallery = torchvision.utils.make_grid(random_img, nrow=5, normalize=True, range=(0, 1))
        fixed_gallery = torchvision.utils.make_grid(fixed_img, nrow=5, normalize=True, range=(0, 1))
        reference_gallery = torchvision.utils.make_grid(reference_img, nrow=5, normalize=True, range=(0, 1))

        log.add_image('random', random_gallery, iteration)
        log.add_image('fixed', fixed_gallery, iteration)
        log.add_image('reference', reference_gallery, iteration)

        iteration_fixed = '{:08d}'.format(iteration)
        torchvision.utils.save_image(random_gallery, '{}/random_{}.png'.format(samples_dir, iteration_fixed))
        torchvision.utils.save_image(fixed_gallery, '{}/fixed_{}.png'.format(samples_dir, iteration_fixed))
        torchvision.utils.save_image(reference_gallery, '{}/reference_{}.png'.format(samples_dir, iteration_fixed))

    # Save models
    if iteration % args.save_models == 0:
        print 'Saving models'
        torch.save(netD, '{}/discriminator_{}.torch'.format(models_dir, iteration))
        torch.save(netG, '{}/generator_{}.torch'.format(models_dir, iteration))
