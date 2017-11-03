import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

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

        self.activation = activation

        self.enc1 = nn.Conv2d(1,64,4,2,1) #8x14x14
        self.enc1_bn = nn.BatchNorm2d(64)
        self.enc2 = nn.Conv2d(64,64,4,2,2) #16x8x8
        self.enc2_bn = nn.BatchNorm2d(64)
        self.enc3 = nn.Conv2d(64,128,4,2,1) #32x4x4
        self.enc3_bn = nn.BatchNorm2d(128)
        self.enc4 = nn.Conv2d(128,100,4,2,0) #32x1x1
        self.enc4_bn = nn.BatchNorm2d(100)
        self.enc5 = nn.Linear(100*self.n_digits, 256)
        self.enc6 = nn.Linear(256, 256)
        self.enc_mu = nn.Linear(256, self.latent)
        self.enc_sigma = nn.Linear(256, self.latent)

        self.dec1 = nn.Linear(self.latent, 128)
        self.dec2 = nn.Linear(128, 128)
        self.dec3 = nn.Linear(128, 100*n_digits)
        self.dec4 = nn.ConvTranspose2d(100,128,4,1,0) #32x4x4
        self.dec4_bn = nn.BatchNorm2d(128)
        self.dec5 = nn.ConvTranspose2d(128,128,4,2,1) #16x8x8
        self.dec5_bn = nn.BatchNorm2d(128)
        self.dec6 = nn.ConvTranspose2d(128,64,4,2,2) #8x14x14
        self.dec6_bn = nn.BatchNorm2d(64)
        self.dec7 = nn.ConvTranspose2d(64,64,4,2,1) #4x28x28
        self.dec7_bn = nn.BatchNorm2d(64)
        self.dec_mu = nn.ConvTranspose2d(64,1,1) #1x28x28
        self.dec_sigma = nn.ConvTranspose2d(64,1,1) #1x28x28


    def encode(self, x):
        z = self.activation(self.enc1_bn(self.enc1(x.view(-1,1,28,28))))
        z = self.activation(self.enc2_bn(self.enc2(z)))
        z = self.activation(self.enc3_bn(self.enc3(z)))
        z = self.activation(self.enc4_bn(self.enc4(z))).view(-1,self.n_digits*100)
        z = self.activation(self.enc5(z))
        z = self.activation(self.enc6(z))
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

        x = self.activation(self.dec4_bn(self.dec4(z.view(-1,100,1,1))))
        x = self.activation(self.dec5_bn(self.dec5(x)))
        x = self.activation(self.dec6_bn(self.dec6(x)))
        x = self.activation(self.dec7_bn(self.dec7(x)))
        x_mu = F.sigmoid(self.dec_mu(x))
        #x_logvar = self.dec_sigma(x)

        return x_mu.view(-1,self.n_digits,28,28)#, x_logvar.view(-1,self.n_digits,28,28)


    def forward(self, x):
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_mu = self.decode(z)
        return x_mu, z_mu, z_logvar

    def generate(self, batch_size=100, noise=None, volatile=False, use_cuda=True):
        if noise is None:
            noise = torch.randn(batch_size, self.latent)
            if use_cuda:
                noise = noise.cuda()
            noise = Variable(noise, volatile=volatile)
        x_mu = self.decode(noise)
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
            [32, 4, 2, 1],
            # 14 x 14 - 128
            [64, 4, 2, 2],
            # 8 x 8 - 128
            [128, 4, 2, 1],
            # 4 x 4 - 128
        ]

        self.decoder_sizes = [
            [128, None, None, None],
            # 4 x 4 - 128
            [64, 4, 2, 1],
            # 8 x 8 - 128
            [32, 4, 2, 2],
            # 14 x 14 - 128
            [16, 4, 2, 1],
            # 28 x 28 - n_digits
        ]

        # Build encoder
        self.encoder_modules = build_cnn(self.encoder_sizes,
                                         batchnorm,
                                         deconv=False)
        self.encoder_modules == [nn.ReLU(True)]
        self.encoder = nn.Sequential(*self.encoder_modules)

        # Mean and covariance from encoder output
        #self.to_mean = nn.Conv2d(self.encoder_sizes[-1][0], self.latent, 1, 1, 0)
        #self.to_cov = nn.Conv2d(self.encoder_sizes[-1][0], self.latent, 1, 1, 0)
        self.enc_1 = nn.Linear(128*4*4, 100)
        self.enc_2 = nn.Linear(100, 100)
        self.to_mean = nn.Linear(100, self.latent)
        self.to_cov = nn.Linear(100, self.latent)
        # 1 x 1 - latent

        # Build decoder
        self.decoder_modules = build_cnn(self.decoder_sizes,
                                 batchnorm=batchnorm,
                                 deconv=True)
        # ReLU is better than Sigmoid because data contains mostly 0
        # so everything will be centered but positive
        self.decoder_modules += [nn.ReLU(True)]
        self.decoder = nn.Sequential(*self.decoder_modules)
        self.dec_1 = nn.Linear(self.latent, 128*4*4)
        self.dec_mu = nn.Conv2d(self.decoder_sizes[-1][0], self.n_digits, 1,1,0)

    def encode(self, x):
        h = self.encoder(x)
        h = F.relu(self.enc_1(h.view(-1, 128*4*4)))
        h = F.relu(self.enc_2(h))
        return self.to_mean(h).view(-1,self.latent,1,1), self.to_cov(h).view(-1,self.latent,1,1)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        x = F.relu(self.dec_1(z.view(-1,self.latent)))
        x = self.decoder(x.view(-1,128,4,4))
        return self.dec_mu(x)

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
