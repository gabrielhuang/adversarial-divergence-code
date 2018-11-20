from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_dim, filters, channels):
        super(Generator, self).__init__()

        nz = latent_dim
        ngf = filters
        nc = channels

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self, filters, channels):
        super(Discriminator, self).__init__()

        nc = channels
        ndf = filters

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)

        return output.view(-1, 1).squeeze(1)


class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class MnistGenerator(nn.Module):
    def __init__(self, latent_dim=100, filters=64):
        super(MnistGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.filters = filters

        self.main = nn.Sequential(
            # latent:1x1
            nn.ConvTranspose2d(latent_dim, 4*filters, 4, 1, 0),
            nn.ReLU(True),
            # 256:4x4
            nn.ConvTranspose2d(4*filters, 2*filters, 4, 2, 1),
            nn.ReLU(True),
            # 128:8x8
            nn.ConvTranspose2d(2*filters, filters, 4, 2, 2),
            nn.ReLU(True),
            # 64:14x14
            nn.ConvTranspose2d(filters, 1, 4, 2, 1),
            nn.Tanh()
            # 1:28x28
        )

    def forward(self, input):
        output = self.main(input)
        return output


class MnistGeneratorBN(nn.Module):
    def __init__(self, latent_dim=100, filters=64):
        super(MnistGeneratorBN, self).__init__()

        self.latent_dim = latent_dim
        self.filters = filters

        self.main = nn.Sequential(
            # latent:1x1
            nn.ConvTranspose2d(latent_dim, 4*filters, 4, 1, 0),
            nn.BatchNorm2d(4*filters),
            nn.ReLU(True),
            # 256:4x4
            nn.ConvTranspose2d(4*filters, 2*filters, 4, 2, 1),
            nn.BatchNorm2d(2*filters),
            nn.ReLU(True),
            # 128:8x8
            nn.ConvTranspose2d(2*filters, filters, 4, 2, 2),
            nn.BatchNorm2d(filters),
            nn.ReLU(True),
            # 64:14x14
            nn.ConvTranspose2d(filters, 1, 4, 2, 1),
            nn.Tanh()
            # 1:28x28
        )

    def forward(self, input):
        output = self.main(input)
        return output


class MnistDiscriminator(nn.Module):
    def __init__(self, latent_dim=100, filters=64):
        super(MnistDiscriminator, self).__init__()

        self.latent_dim = latent_dim
        self.filters = filters

        self.main = nn.Sequential(
            # 1:28x28
            nn.Conv2d(1, filters, 4, 2, 1),
            nn.ReLU(True),
            # 64:14x14
            nn.Conv2d(filters, 2*filters, 4, 2, 2),
            nn.ReLU(True),
            # 128:8x8
            nn.Conv2d(2*filters, 4*filters, 4, 2, 1),
            nn.ReLU(True),
            # 256:4x4
            nn.Conv2d(4*filters, latent_dim, 4, 1, 0),
            nn.ReLU(True),
            # latent:1x1
            ReshapeLayer([-1, latent_dim]),
            nn.Linear(latent_dim, 1),
            # 1
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.sigmoid(self.main(input))
        return output

    def get_logits(self, input):
        output = self.main(input)
        return output

