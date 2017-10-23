import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


#########################
# Networks
########################


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
