import os
import time
from tqdm import tqdm
import argparse
import json
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.autograd import grad
from tensorboardX import SummaryWriter  # install with pip install git+https://github.com/lanpa/tensorboard-pytorch
from models_gan import ImageDiscriminator, ImageGenerator, UnconstrainedImageDiscriminator, UnconstrainedImageGenerator, SemiSupervisedImageDiscriminator
from hyperplane_dataset import get_full_train_test, HyperplaneImageDataset

parser = argparse.ArgumentParser(description='Train GAN with visual hyperplane')

# general learning
parser.add_argument('--iterations', default=100000, type=int, help='iterations')
parser.add_argument('--batch_size', default=64, type=int, help='minibatch size')
parser.add_argument('--threads', default=8, type=int, help='number of threads for data loading')
parser.add_argument('--logdir', required=True, help='where to log samples and models')
parser.add_argument('--save-samples', default=100, type=int, help='save samples every')
parser.add_argument('--save-models', default=1000, type=int, help='save models every')
parser.add_argument('--log-every', default=100, type=int, help='log every N iterations')
parser.add_argument('--use-cuda', default=1, type=int, help='use cuda')
parser.add_argument('--validate-every', default=20, type=int, help='validate every N iterations')
parser.add_argument('--generate-samples', default=64, type=int, help='generate N samples')
parser.add_argument('--random-seed', default=1234, type=int, help='random seed')
parser.add_argument('--mnist', default='data', help='folder where MNIST is/will be downloaded')
parser.add_argument('--sample-rows', default=10, type=int, help='how many samples in tensorboard')

# task specific
parser.add_argument('--amount', default=25, type=int, help='target to sum up to')
parser.add_argument('--nb_digits', default=5, type=int, help='how many digits per sequence')

# WGAN-GP specific
parser.add_argument('-p', '--penalty', default=10., type=float, help='gradient penalty')
parser.add_argument('--double-sided', default=0, type=int, help='use double sided penalty vs single sided')
parser.add_argument('--glr', default=1e-4, type=float, help='generator learning rate')
parser.add_argument('--dlr', default=1e-4, type=float, help='discriminator learning rate')
parser.add_argument('--batchnorm', default=1, type=int, help='whether to use batchnorm')
parser.add_argument('--latent-local', default=10, type=int, help='local latent dimensions for each image')
parser.add_argument('--latent-global', default=64, type=int, help='global latent dimensions')
parser.add_argument('--critic-iterations', default=5, type=int, help='number of critic iterations')
parser.add_argument('--model-generator', default='constrained', help='constrained|unconstrained')
parser.add_argument('--model-discriminator', default='constrained', help='constrained|unconstrained|semi')
parser.add_argument('--unconstrained-size', default=32, type=int, help='size of disc/gen in unconstrained case')

args = parser.parse_args()
assert args.model_generator in ['constrained', 'unconstrained']
assert args.model_discriminator in ['constrained', 'unconstrained', 'semi']
short = {'constrained': 'C', 'unconstrained': 'U', 'semi': 'S'}
print args

date = time.strftime('%Y-%m-%d.%H%M')
run_dir = '{}/gan-d{}g{}-{}'.format(args.logdir, short[args.model_discriminator],
                                short[args.model_generator], date)
# Create models dir if deosnt exist
models_dir = '{}/models'.format(run_dir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

with open('{}/args.json'.format(run_dir), 'wb') as f:
    json.dump(vars(args), f, indent=4)


def view_samples(data, max_samples=None):
    '''
    reshape from (batch, args.digits, 28, 28) to (batch*args.digits, 1, 28, 28)
    '''
    if max_samples:
        data = data[:min(max_samples, len(data))]
    size = data.size()
    return data.view(-1, 1, size[-2], size[-1])


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
    zeros = Variable(torch.zeros(len(real_data)))  # has to be variable
    if cuda:
        ones = ones.cuda()
        zeros = zeros.cuda()

    gradients = grad(outputs=D_interpolates, inputs=interpolates_flat,
                     grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0]

    if double_sided:
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    else:
        gradient_penalty = torch.max((gradients*gradients).sum(1)-1, zeros)
        gradient_penalty = gradient_penalty.mean()

    return gradient_penalty


full, train, test = get_full_train_test(args.amount, range(10), args.nb_digits,
                                        one_hot=False, validation=0.8, seed=args.random_seed)
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
if args.model_discriminator == 'unconstrained':
    netD = UnconstrainedImageDiscriminator(args.nb_digits, args.unconstrained_size)
elif args.model_discriminator == 'constrained':
    netD = ImageDiscriminator(args.nb_digits)
elif args.model_discriminator == 'semi':
    netD = SemiSupervisedImageDiscriminator(args.nb_digits)
if args.model_generator == 'unconstrained':
    netG = UnconstrainedImageGenerator(args.latent_global, args.nb_digits, args.unconstrained_size)
else:
    netG = ImageGenerator(args.latent_global, args.latent_local, args.nb_digits)

if args.use_cuda:
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

nll_criterion = nn.NLLLoss()

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
        real_data, real_labels = train_iter.next()
        if args.use_cuda:
            real_data = real_data.cuda()
            real_labels = real_labels.cuda()
        real_data = Variable(real_data)
        real_labels = Variable(real_labels)
        if len(real_data) != args.batch_size:
            # the last batch might be smaller, skip it
            continue
        D_real = netD(real_data).mean()

        # Fake data
        # volatile: do not compute gradient for netG
        # stop gradient at fake_data
        fake_data = Variable(netG.generate(args.batch_size, use_cuda=args.use_cuda, volatile=True).data)
        D_fake = netD(fake_data).mean()

        # Costs
        gradient_penalty = args.penalty * get_gradient_penalty(
            netD, real_data.data, fake_data.data, double_sided=args.double_sided, cuda=args.use_cuda)
        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake

        if args.model_discriminator == 'semi':
            #import pdb pdb.set_trace()
            prediction = netD.get_prediction()
            ground_truth = real_labels.view(-1)
            classification_cost = nll_criterion(prediction, ground_truth)
            D_cost += classification_cost
            indices = np.argmax(prediction.cpu().numpy(), axis=1)
            accuracy = (indices == ground_truth.data.cpu().numpy()).mean()
            log.add_scalar('semiSupervisedCost', classification_cost.cpu().data.numpy(), iteration)
            log.add_scalar('semiSupervisedAccuracy', classification_cost.cpu().data.numpy(), iteration)

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
    fake_data = netG.generate(args.batch_size, use_cuda=args.use_cuda, volatile=False)
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

    # Reconstructions
    if iteration % args.save_samples == 0:
        # Generate samples
        samples = netG.generate(args.generate_samples, use_cuda=args.use_cuda)

        # Process and log
        view_train = view_samples(real_data.data, args.sample_rows)
        view_gen = view_samples(samples.data, args.sample_rows)

        gallery_train = torchvision.utils.make_grid(view_train, nrow=args.nb_digits, normalize=True, range=(0, 1))
        gallery_gen = torchvision.utils.make_grid(view_gen, nrow=args.nb_digits, normalize=True, range=(0, 1))

        log.add_image('train', gallery_train, iteration)
        log.add_image('generation', gallery_gen, iteration)

    # Save models
    if iteration % args.save_models == 0:
        frame_str = '{:08d}'.format(iteration)
        print 'Saving models'
        torch.save(netD, '{}/discriminator_{}.torch'.format(models_dir, iteration))
        torch.save(netG, '{}/generator_{}.torch'.format(models_dir, iteration))
