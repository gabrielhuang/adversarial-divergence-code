import torch
from torch import nn
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
    def __init__(self, latent, resolution, batchnorm=True):
        super(VAE, self).__init__()

        self.latent = latent
        self.resolution = resolution
        self.batchnorm = batchnorm

        self.encoder_sizes = [
            [1, None, None, None],
            # 512 x 512 - 1
            [16, 4, 2, 1],
            # 256 x 256 - 16
            [16, 4, 2, 1],
            # 128 x 128 - 16
            [32, 4, 2, 1],
            # 64 x 64 - 32
            [64, 4, 2, 1],
            # 32 x 32 - 64
            [128, 4, 2, 1],
            # 16 x 16 - 128
            [128, 4, 2, 1],
            # 8 x 8 - 128
            [128, 4, 2, 1],
            # 4 x 4 - 128
            [128, 4, 1, 0],
            # 1 x 1 - 128
        ]

        self.decoder_sizes = [
            [self.latent, None, None, None],
            # 1 x 1 - latent
            [128, 4, 1, 0],
            # 4 x 4 - 128
            [128, 4, 2, 1],
            # 8 x 8 - 128
            [128, 4, 2, 1],
            # 16 x 16 - 128
            [64, 4, 2, 1],
            # 32 x 32 - 64
            [32, 4, 2, 1],
            # 64 x 64 - 32
            [16, 4, 2, 1],
            # 128 x 128 - 16
            [16, 4, 2, 1],
            # 256 x 256 - 16
            [1, 4, 2, 1],
            # 512 x 512 - 1
        ]

        # Truncate VAE for smaller sizes
        if resolution == 256:
            self.encoder_sizes = self.encoder_sizes[1:]
            self.decoder_sizes = self.decoder_sizes[:-1]
        elif resolution == 128:
            self.encoder_sizes = self.encoder_sizes[2:]
            self.decoder_sizes = self.decoder_sizes[:-2]
        elif resolution == 64:
            self.encoder_sizes = self.encoder_sizes[3:]
            self.decoder_sizes = self.decoder_sizes[:-3]
        elif resolution == 32:
            self.encoder_sizes = self.encoder_sizes[4:]
            self.decoder_sizes = self.decoder_sizes[:-4]
        # Make input and output channels consistent with data
        self.encoder_sizes[0][0] = 1  # set input channels to 1
        self.decoder_sizes[-1][0] = 1  # set output channels to 1

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
        self.decoder_modules += [nn.ReLU()]
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

    def generate(self, batch_size=100, noise=None, volatile=False):
        '''
        n is either integer or latent variable
        '''
        if noise is None:
            noise = torch.randn(batch_size, self.latent, 1, 1)
            noise = noise.cuda()
            noise = Variable(noise, volatile=volatile)
        samples = self.decode(noise)
        return samples
