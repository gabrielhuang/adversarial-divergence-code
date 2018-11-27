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


class MNISTNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)



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
        self.conv_sizes = [
            [1, None, None, None],
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
        self.conv = nn.Sequential(
            *build_cnn(self.conv_sizes, batchnorm=False, deconv=False)
        )
        self.mlp_sizes = [
            [128*self.nb_digits, 128],
            [128, 128],
            [128, 1]
        ]
        self.mlp_modules = build_mlp(self.mlp_sizes, batchnorm=False)[:-1]
        self.mlp = nn.Sequential(
            *self.mlp_modules
        )

    def forward(self, input):
        out = input.view(-1, 1, 28, 28)
        out = self.conv(out)
        out = out.view(len(input), -1)
        out = self.mlp(out)
        return out.view(-1, 1)


class UnconstrainedImageGenerator(nn.Module):
    def __init__(self, latent, nb_digits, size=32):
        nn.Module.__init__(self)
        self.latent = latent
        self.nb_digits = nb_digits
        self.size = size

        self.deconv_sizes = [
            [latent, None, None, None],
            # 1 x 1 - latent
            [size*8, 4, 1, 0],
            # 4 x 4 - 128
            [size*4, 4, 2, 1],
            # 8 x 8 - 128
            [size*2, 4, 2, 2],
            # 14 x 14 - 64
            [self.nb_digits, 4, 2, 1]
            # 28 x 18 - 1
        ]
        self.deconv = nn.Sequential(
            *build_cnn(self.deconv_sizes, batchnorm=True, deconv=True)
        )

    def forward(self, input):
        out = input.view(-1, self.latent, 1, 1)
        out = self.deconv(out)
        return out

    def generate(self, batch_size, use_cuda, volatile=True):
        noise = torch.randn(batch_size, self.latent)
        if use_cuda:
            noise = noise.cuda()
        noise = Variable(noise, volatile=volatile)
        samples = self(noise)
        return samples


class UnconstrainedImageDiscriminator(nn.Module):
    def __init__(self, nb_digits, size=32):
        nn.Module.__init__(self)
        self.nb_digits = nb_digits
        self.size = size
        self.sizes = [
            [nb_digits, None, None, None],
            # 28 x 28 - nb_digits
            [size*1, 4, 2, 1],
            # 14 x 14 - size
            [size*2, 4, 2, 2],
            # 8 x 8 - size*2
            [size*4, 4, 2, 1],
            # 4 x 4 - size*4
            [size*8, 4, 1, 0],
            # 1 x 1 - size*8
            [1, 1, 1, 0]
            # 1 x 1 - 1
        ]
        self.main = nn.Sequential(
            *build_cnn(self.sizes, batchnorm=False, deconv=False)
        )

    def forward(self, input):
        out = self.main(input)
        return out.view(-1, 1)


class SemiSupervisedImageDiscriminator(nn.Module):
    def __init__(self, nb_digits, size=32):
        nn.Module.__init__(self)
        self.nb_digits = nb_digits
        self.size = size
        self.sizes = [
            [nb_digits, None, None, None],
            # 28 x 28 - nb_digits
            [size*1, 4, 2, 1],
            # 14 x 14 - size
            [size*2, 4, 2, 2],
            # 8 x 8 - size*2
            [size*4, 4, 2, 1],
            # 4 x 4 - size*4
            [size*8, 4, 1, 0],
            # 1 x 1 - size*8
        ]
        self.main = nn.Sequential(
            *build_cnn(self.sizes, batchnorm=False, deconv=False)
        )
        self.discriminator_output = nn.Linear(size*8, 1)
        self.classifier_output = nn.Linear(size*8, nb_digits*10)

    def forward(self, input):
        out = self.main(input)
        out = out.view(len(input), -1)
        d_out = self.discriminator_output(out)
        c_out = self.classifier_output(out)
        c_out = c_out.view(-1, 10)
        c_out = F.log_softmax(c_out)
        self.c_out = c_out
        return d_out.view(-1, 1)

    def get_prediction(self):
        return self.c_out


if __name__ == '__main__':
    import numpy as np
    input=Variable(torch.Tensor(np.random.uniform(size=(2, 3, 28, 28))))
    d = ImageDiscriminator(3)
    d(input)

    ud = UnconstrainedImageDiscriminator(3)
    d(input)
