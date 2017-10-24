import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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


class VAE(nn.Module):
    def __init__(self, n_digits, latent=50, batchnorm=True, activation=F.relu):
        super(VAE, self).__init__()
        self.latent = latent
        self.n_digits = n_digits

        if batchnorm:
            self.activation = lambda x: F.batch_norm(activation(x))
        else:
            self.activation = self.activation

        self.enc1 = nn.Conv2d(1,8,4,2,1) #8x14x14
        self.enc2 = nn.Conv2d(8,16,4,2,2) #16x8x8
        self.enc3 = nn.Conv2d(16,32,4,2,1) #32x4x4
        self.enc4 = nn.Conv2d(32,32,4,2,0) #32x1x1
        self.enc_mu = nn.Linear(200, self.latent)
        self.enc_sigma = nn.Linear(200, self.latent)

        self.dec1 = nn.Linear(self.latent, 200)
        self.dec2 = nn.Linear(200, 200)
        self.dec3 = nn.Linear(200, 32*n_digits)
        self.dec4 = nn.ConvTranspose2d(32,32,4,1,0) #32x4x4
        self.dec5 = nn.ConvTranspose2d(32,16,4,2,1) #16x8x8
        self.dec6 = nn.ConvTranspose2d(16,8,4,2,2) #8x14x14
        self.dec7 = nn.ConvTranspose2d(8,4,4,2,1) #4x28x28
        self.dec_mu = nn.ConvTranspose2d(4,1,1) #1x28x28
        self.dec_sigma = nn.ConvTranspose2d(4,1,1) #1x28x28


    def encode(self, x):
        z = self.activation(self.enc1(x.view(-1,1,28,28)))
        z = self.activation(self.enc2(z))
        z = self.activation(self.enc3(z))
        z = self.activation(self.enc4(z)).view(-1,self.n_digits*32)
        z_mu = self.enc_mu(z)
        z_sigma = self.enc_sigma(z)

        return z_mu, z_sigma

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        z = self.activation(self.dec1(z))
        z = self.activation(self.dec2(z))
        z = self.activation(self.dec3(z))

        x = self.activation(self.dec4(x.view(-1,1,28,28)))
        x = self.activation(self.dec5(x))
        x = self.activation(self.dec6(x))
        x = self.activation(self.dec7(x))
        x_mu = self.dec_mu(x)
        x_logvar = self.dec_sigma(x)

        return x_mu.view(-1,self.n_digits,28,28), x_logvar.view(-1,self.n_digits,28,28)


    def forward(self, x):
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_mu, x_logvar = self.decode(z)
        return x_mu, x_logvar, z_mu, z_logvar

    def generate(self, batch_size=100, noise=None, volatile=False, use_cuda=True):
        if noise is None:
            noise = torch.randn(batch_size, self.latent)
            if use_cuda:
                noise = noise.cuda()
            noise = Variable(noise, volatile=volatile)
        x_mu, x_logvar = self.decode(noise)
        return x_mu


class UnconstrainedVAE(nn.Module):
    def __init__(self, latent, n_digits, batchnorm=True):
        nn.Module.__init__(self)

        self.latent = latent
        self.n_digits = n_digits
        self.batchnorm = batchnorm

        # Build network sizes
        self.encoder_sizes = [
            [n_digits, None, None, None],
            # 28 x 28 - n_digits
            [64, 4, 2, 1],
            # 14 x 14 - 128
            [128, 4, 2, 2],
            # 8 x 8 - 128
            [128, 4, 2, 1],
            # 4 x 4 - 128
            [128, 4, 1, 0],
            # 1 x 1 - 128
        ]

        self.decoder_sizes = [
            [latent, None, None, None],
            # 1 x 1 - latent
            [128, 4, 1, 0],
            # 4 x 4 - 128
            [128, 4, 2, 1],
            # 8 x 8 - 128
            [64, 4, 2, 2],
            # 14 x 14 - 128
            [n_digits, 4, 2, 1],
            # 28 x 28 - n_digits
        ]

        # Build encoder
        self.encoder_modules = build_cnn(self.encoder_sizes,
                                         batchnorm,
                                         deconv=False)
        self.encoder_modules == [nn.ReLU(True)]
        self.encoder = nn.Sequential(*self.encoder_modules)

        # Mean and covariance from encoder output
        self.to_mean = nn.Conv2d(self.encoder_sizes[-1][0], self.latent, 1, 1, 0)
        self.to_cov = nn.Conv2d(self.encoder_sizes[-1][0], self.latent, 1, 1, 0)
        # 1 x 1 - latent

        # Build decoder
        self.decoder_modules = build_cnn(self.decoder_sizes,
                                 batchnorm=batchnorm,
                                 deconv=True)
        # ReLU is better than Sigmoid because data contains mostly 0
        # so everything will be centered but positive
        self.decoder_modules += [nn.ReLU(True)]
        self.decoder = nn.Sequential(*self.decoder_modules)

    def encode(self, x):
        h = self.encoder(x)
        return self.to_mean(h), self.to_cov(h)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def generate(self, batch_size=100, noise=None, volatile=False, use_cuda=True):
        if noise is None:
            noise = torch.randn(batch_size, self.latent, 1, 1)
            if use_cuda:
                noise = noise.cuda()
            noise = Variable(noise, volatile=volatile)
        samples = self.decode(noise)
        return samples


class Generator(nn.Module):

    def __init__(self, latent, resolution, batchnorm=True):
        nn.Module.__init__(self)
        self.latent = latent
        self.resolution = resolution
        self.batchnorm = batchnorm

        __ , self.generator_sizes  = get_sizes(latent, resolution)

        self.modules = build_cnn(self.generator_sizes,
                              batchnorm,
                              deconv=True)
        self.modules += [nn.ReLU(True)]

        self.main = nn.Sequential(*self.modules)

    def forward(self, input):
        out = self.main(input)
        return out

    def generate(self, batch_size=100, noise=None, volatile=False, use_cuda=True):
        if noise is None:
            noise = torch.randn(batch_size, self.latent, 1, 1)
            if use_cuda:
                noise = noise.cuda()
            noise = Variable(noise, volatile=volatile)
        samples = self(noise)
        return samples


class Discriminator(nn.Module):

    def __init__(self, latent, resolution, batchnorm=False):
        nn.Module.__init__(self)
        self.latent = latent
        self.resolution = resolution
        self.batchnorm = batchnorm

        self.discriminator_sizes, __ = get_sizes(latent, resolution)

        self.modules = build_cnn(self.discriminator_sizes,
                              batchnorm,
                              deconv=False)
        self.modules += [nn.ReLU(True)]
        self.modules += [nn.Conv2d(self.discriminator_sizes[-1][0], 1, 1, 1, 0)]

        self.main = nn.Sequential(*self.modules)

    def forward(self, input):
        out = self.main(input)
        return out.view(-1, 1)  # scalar output

