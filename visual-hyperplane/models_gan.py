import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


#########################
# Networks
########################

def build_cnn(sizes, batchnorm=True, deconv=False):
    '''
    sizes: (output channel, kernel_size, stride, padding)
    '''
    modules = []
    for i, (out_ch, kernel, stride, padding) in enumerate(sizes):
        if i == 0:
            continue
        in_ch = sizes[i-1][0]
        if deconv:
            modules.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel, stride, padding))
        else:
            modules.append(
                nn.Conv2d(in_ch, out_ch, kernel, stride, padding))
        if batchnorm:
            modules.append(
                nn.BatchNorm2d(out_ch))
        if i != len(sizes)-1:
            modules.append(
                nn.ReLU(True))
    return modules


def build_mlp(sizes, batchnorm=True):
    modules = []
    for (in_dim, out_dim) in sizes:
        modules.append(nn.Linear(in_dim, out_dim))
        if batchnorm:
            modules.append(nn.BatchNorm1d(out_dim))
        modules.append(nn.ReLU(True))
    return modules


def gumbel_softmax_sampler(logits, tau, use_cuda):
    noise = torch.rand(logits.size())
    if use_cuda:
        noise = noise.cuda()
    noise.add_(1e-9).log_().neg_()
    noise.add_(1e-9).log_().neg_()
    noise = Variable(noise)
    x = (logits + noise) / tau
    x = F.softmax(x.view(logits.size(0), -1))
    return x.view_as(logits)


class DigitGenerator(nn.Module):

    def __init__(self, nb_digits, dim, latent, mode='softmax'):
        nn.Module.__init__(self)
        self.latent = latent
        self.nb_digits = nb_digits
        self.dim = dim
        self.mode = mode
        assert mode in ['softmax', 'gumbel', 'hard']
        N = nb_digits * dim
        self.main = nn.Sequential(
            # latent channels
            nn.Linear(latent, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            # 128 channels
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            # 128 channels
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            # 64 channels
            nn.Linear(64, N),
            nn.BatchNorm1d(N)
            # NB_DIGITS*DIM channels
        )

    def forward(self, input, use_cuda, tau=None):
        out = self.main(input)
        if self.mode == 'gumbel':
            if tau:
                logits = out.view(-1, self.dim)
                out = gumbel_softmax_sampler(logits, tau, use_cuda=use_cuda)
            else:
                raise ValueError('missing temperature (tau= None)')
        elif self.mode == 'softmax':
            logits = out.view(-1, self.dim)
            out = F.softmax(logits)
        elif self.mode == 'hard':
            out = F.relu(out)
        return out.view(-1, self.nb_digits, self.dim)

    def generate(self, batch_size, use_cuda, tau=None, volatile=True):
        noise = torch.randn(batch_size, self.latent)
        if use_cuda:
            noise = noise.cuda()
        noise = Variable(noise, volatile=volatile)
        if self.mode == 'gumbel':
            samples = self(noise, use_cuda, tau=tau)
        else:
            samples = self(noise, use_cuda)
        return samples


class DigitDiscriminator(nn.Module):

    def __init__(self, nb_digits, dim):
        nn.Module.__init__(self)
        self.nb_digits = nb_digits
        self.dim = dim
        N = nb_digits * dim
        self.main = nn.Sequential(
            # NB_DIGITS*DIM channels
            nn.Linear(N, 128),
            nn.ReLU(True),
            # 128 channels
            nn.Linear(128, 128),
            nn.ReLU(True),
            # 128 channels
            nn.Linear(128, 64),
            nn.ReLU(True),
            # 64 channels
            nn.Linear(64, 1),
            # 1 channel
        )

    def forward(self, input):
        N = self.nb_digits * self.dim
        out = input.view(-1, N)
        out = self.main(out)
        return out.view(-1, 1)


class ImageGenerator(nn.Module):
    def __init__(self, latent_global, latent_visual, nb_digits):
        nn.Module.__init__(self)
        self.nb_digits = nb_digits
        self.latent_global = latent_global
        self.latent_visual = latent_visual

        self.latent_sizes = [
            [latent_global, 128],
            [128, 128],
            [128, nb_digits * latent_visual]
        ]
        self.deconv_sizes = [
            [latent_visual, None, None, None],
            # 1 x 1 - latent
            [128, 4, 1, 0],
            # 4 x 4 - 128
            [128, 4, 2, 1],
            # 8 x 8 - 128
            [64, 4, 2, 2],
            # 14 x 14 - 64
            [1, 4, 2, 1]
            # 28 x 18 - 1
        ]
        self.get_latents = nn.Sequential(
            *build_mlp(self.latent_sizes, batchnorm=True)
        )
        self.get_images = nn.Sequential(
            *build_cnn(self.deconv_sizes, batchnorm=True, deconv=True)
        )

    def forward(self, input):
        latent_vars = self.get_latents(input)
        latent_vars = latent_vars.view(-1, self.latent_visual, 1, 1)
        images = self.get_images(latent_vars)
        images = images.view(-1, self.nb_digits, 28, 28)
        return images

    def generate(self, batch_size, use_cuda, volatile=True):
        noise = torch.randn(batch_size, self.latent_global)
        if use_cuda:
            noise = noise.cuda()
        noise = Variable(noise, volatile=volatile)
        samples = self(noise)
        return samples


class ImageDiscriminator(nn.Module):
    def __init__(self, nb_digits):
        nn.Module.__init__(self)
        self.nb_digits = nb_digits
        self.sizes = [
            [nb_digits, None, None, None],
            # 28 x 28 - nb_digits
            [32, 4, 2, 1],
            # 14 x 14 - 32
            [32, 4, 2, 2],
            # 8 x 8 - 32
            [64, 4, 2, 1],
            # 4 x 4 - 64
            [128, 4, 1, 0]
            # 1 x 1 - 128
        ]
        self.main = nn.Sequential(
            *build_cnn(self.sizes, batchnorm=False, deconv=False)
        )

    def forward(self, input):
        out = self.main(input)
        return out.view(-1, 1)
