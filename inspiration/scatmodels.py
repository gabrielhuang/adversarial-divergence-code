import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from torch.autograd import Variable


class ImplicitGenerator(nn.Module):
    def __init__(self, latent):
        nn.Module.__init__(self)
        self.latent = latent

    @abstractmethod
    def forward(self, input):
        pass

    def generate(self, batch_size, volatile=False):
        noise = torch.randn(batch_size, self.latent)
        noise = noise.cuda()
        noise = Variable(noise, volatile=volatile)
        samples = self(noise)
        return samples


class MnistGenerator(ImplicitGenerator):
    '''
    See http://pytorch.org/docs/master/nn.html#torch.nn.ConvTranspose2d
    for shapes using ConvTranspose2d
    '''
    SCAT_CHANNELS = 81
    SCAT_HEIGHT = 7
    SCAT_WIDTH = 7

    def __init__(self, latent):
        ImplicitGenerator.__init__(self, latent)

        N = self.SCAT_CHANNELS*self.SCAT_HEIGHT*self.SCAT_WIDTH
        self.dense1 = nn.Linear(latent, N/4)
        self.dense2 = nn.Linear(N/4, N/2)
        self.dense3 = nn.Linear(N/2, N)

    def forward(self, input):
        out = F.relu(self.dense1(input))
        out = F.relu(self.dense2(out))
        out = self.dense3(out)
        return out.view(-1, 1, self.SCAT_CHANNELS, self.SCAT_HEIGHT, self.SCAT_WIDTH)


class MnistDiscriminator(nn.Module):
    SCAT_CHANNELS = 81
    SCAT_HEIGHT = 7
    SCAT_WIDTH = 7

    def __init__(self):
        nn.Module.__init__(self)

        N = self.SCAT_CHANNELS*self.SCAT_HEIGHT*self.SCAT_WIDTH
        self.dense1 = nn.Linear(N, N/2)
        self.dense2 = nn.Linear(N/2, N/4)
        self.dense3 = nn.Linear(N/4, 1)

    def forward(self, input):
        out = input.view(-1, self.SCAT_CHANNELS*self.SCAT_HEIGHT*self.SCAT_WIDTH)
        out = F.relu(self.dense1(out))
        out = F.relu(self.dense2(out))
        out = self.dense3(out)
        return out.view(-1, 1)


class LSUNGenerator(MnistGenerator):
    IMAGE_CHANNELS = 3
    SCAT_CHANNELS = 81
    SCAT_HEIGHT = 16
    SCAT_WIDTH = 16
    HIDDEN = 512

    def __init__(self, latent):
        ImplicitGenerator.__init__(self, latent)

        N = self.IMAGE_CHANNELS*self.SCAT_CHANNELS*self.SCAT_HEIGHT*self.SCAT_WIDTH
        self.dense1 = nn.Linear(latent, self.HIDDEN)
        self.dense2 = nn.Linear(self.HIDDEN, self.HIDDEN)
        self.dense3 = nn.Linear(self.HIDDEN, N)

    def forward(self, input):
        out = F.relu(self.dense1(input))
        out = F.relu(self.dense2(out))
        out = self.dense3(out)
        return out.view(-1, self.IMAGE_CHANNELS,
                        self.SCAT_CHANNELS, self.SCAT_HEIGHT, self.SCAT_WIDTH)



class LSUNDiscriminator(MnistDiscriminator):
    IMAGE_CHANNELS = 3
    SCAT_CHANNELS = 81
    SCAT_HEIGHT = 16
    SCAT_WIDTH = 16
    HIDDEN = 512

    def __init__(self):
        nn.Module.__init__(self)

        N = self.IMAGE_CHANNELS*self.SCAT_CHANNELS*self.SCAT_HEIGHT*self.SCAT_WIDTH
        self.dense1 = nn.Linear(N, self.HIDDEN)
        self.dense2 = nn.Linear(self.HIDDEN, self.HIDDEN)
        self.dense3 = nn.Linear(self.HIDDEN, 1)

    def forward(self, input):
        out = input.view(-1, self.IMAGE_CHANNELS*self.SCAT_CHANNELS*self.SCAT_HEIGHT*self.SCAT_WIDTH)
        out = F.relu(self.dense1(out))
        out = F.relu(self.dense2(out))
        out = self.dense3(out)
        return out.view(-1, 1)


class Scat_64_2_Generator(ImplicitGenerator):
    IMAGE_CHANNELS = 3
    SCAT_CHANNELS = 81
    SCAT_HEIGHT = 16
    SCAT_WIDTH = 16

    def __init__(self, latent):
        ImplicitGenerator.__init__(self, latent)

        nc = self.IMAGE_CHANNELS*self.SCAT_CHANNELS

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(latent, nc*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nc*4),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(nc*4, nc*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nc*2),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(nc*2, nc, 4, 2, 1, bias=False),
            # state size. (ngf*2) x 16 x 16
        )

    def forward(self, input):
        out = input.view(-1, self.latent, 1, 1)  # reshape to 1 x 1 image
        out = self.main(out)
        out = out.view(-1, self.IMAGE_CHANNELS, self.SCAT_CHANNELS,
                       self.SCAT_HEIGHT, self.SCAT_WIDTH)
        return out


class Scat_64_2_Discriminator(nn.Module):
    IMAGE_CHANNELS = 3
    SCAT_CHANNELS = 81
    SCAT_HEIGHT = 16
    SCAT_WIDTH = 16

    def __init__(self):
        nn.Module.__init__(self)

        nc = self.IMAGE_CHANNELS*self.SCAT_CHANNELS

        self.main = nn.Sequential(
            # state size. (nc) x 16 x 16
            nn.Conv2d(nc, nc*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nc*2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nc*2) x 8 x 8
            nn.Conv2d(nc*2, nc*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nc*4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nc*4) x 4 x 4
            nn.Conv2d(nc*4, 1, 4, 1, 0, bias=False),
        )

    def forward(self, input):
        out = input.view(-1, self.IMAGE_CHANNELS*self.SCAT_CHANNELS,
                         self.SCAT_HEIGHT, self.SCAT_WIDTH)
        out = self.main(out)
        return out




class Scat_64_2_Generator_2(ImplicitGenerator):
    IMAGE_CHANNELS = 3
    SCAT_CHANNELS = 81
    SCAT_HEIGHT = 16
    SCAT_WIDTH = 16

    def __init__(self, latent):
        ImplicitGenerator.__init__(self, latent)

        nc = self.IMAGE_CHANNELS*self.SCAT_CHANNELS

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent, nc*4, 4, 1, 0),
            nn.BatchNorm2d(nc*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(nc*4, nc*3, 4, 1, 1),
            nn.BatchNorm2d(nc*3),
            nn.ReLU(True),
            nn.ConvTranspose2d(nc*3, nc*3, 3, 1, 1),
            nn.BatchNorm2d(nc*3),
            nn.ReLU(True),
            nn.ConvTranspose2d(nc*3, nc*2, 3, 2, 1),
            nn.BatchNorm2d(nc*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(nc*2, nc, 2, 2, 1),
        )

    def forward(self, input):
        out = input.view(-1, self.latent, 1, 1)  # reshape to 1 x 1 image
        out = self.main(out)
        out = out.view(-1, self.IMAGE_CHANNELS, self.SCAT_CHANNELS,
                       self.SCAT_HEIGHT, self.SCAT_WIDTH)
        return out


class Scat_64_2_Discriminator_2(nn.Module):
    IMAGE_CHANNELS = 3
    SCAT_CHANNELS = 81
    SCAT_HEIGHT = 16
    SCAT_WIDTH = 16

    def __init__(self):
        nn.Module.__init__(self)

        nc = self.IMAGE_CHANNELS*self.SCAT_CHANNELS

        self.main = nn.Sequential(
            nn.Conv2d(nc, nc*2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nc*2, nc*4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nc*4, nc*4, 4, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nc*4, nc*4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nc*4, 1, 3, 1, 0),
        )

    def forward(self, input):
        out = input.view(-1, self.IMAGE_CHANNELS*self.SCAT_CHANNELS,
                         self.SCAT_HEIGHT, self.SCAT_WIDTH)
        out = self.main(out)
        return out



class Scat_64_2_Generator_3(ImplicitGenerator):
    IMAGE_CHANNELS = 3
    SCAT_CHANNELS = 81
    SCAT_HEIGHT = 16
    SCAT_WIDTH = 16

    def __init__(self, latent):
        ImplicitGenerator.__init__(self, latent)

        nc = self.IMAGE_CHANNELS*self.SCAT_CHANNELS

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent, nc*4, 4, 1, 0),
            nn.BatchNorm2d(nc*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(nc*4, nc*3, 4, 2, 1),
            nn.BatchNorm2d(nc*3),
            nn.ReLU(True),
            nn.ConvTranspose2d(nc*3, nc*3, 4, 2, 1),
            nn.BatchNorm2d(nc*3),
            nn.ReLU(True),
            nn.ConvTranspose2d(nc*3, nc*2, 3, 1, 1),
            nn.BatchNorm2d(nc*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(nc*2, nc, 3, 1, 1),
        )

    def forward(self, input):
        out = input.view(-1, self.latent, 1, 1)  # reshape to 1 x 1 image
        out = self.main(out)
        out = out.view(-1, self.IMAGE_CHANNELS, self.SCAT_CHANNELS,
                       self.SCAT_HEIGHT, self.SCAT_WIDTH)
        return out


class Scat_64_2_Discriminator_3(Scat_64_Discriminator_3):
    pass


def get_model(model, latent):
    if model == 'mnist_dense':
        netD = MnistDiscriminator()
        netG = MnistGenerator(latent)
    elif model == 'lsun_dense':
        netD = LSUNDiscriminator()
        netG = LSUNGenerator(latent)
    elif model == 'scat_64_2':
        netD = Scat_64_2_Discriminator()
        netG = Scat_64_2_Generator(latent)
    elif model == 'scat_64_2_2':
        netD = Scat_64_2_Discriminator_2()
        netG = Scat_64_2_Generator_2(latent)
    elif model == 'scat_64_2_3':
        netD = Scat_64_2_Discriminator_3()
        netG = Scat_64_2_Generator_3(latent)
    else:
        raise ValueError('Unkown model {}'.format(model))
    return netD, netG
