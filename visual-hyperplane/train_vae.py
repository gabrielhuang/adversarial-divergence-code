from hyperplane_dataset import HyperplaneCachedDataset
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np

def softmax(input):
    output = Variable(torch.zeros(input.size()))
    for i in range(input.size()[1]):
        output[:,i] = F.log_softmax(input[:,i])
    return output

def nll_loss(x, x_pred):
    output = Variable(torch.zeros(1))
    for i in range(x.size()[1]):
        output += F.nll_loss(x_pred[:,i], x[:,i]).float()
    return output

batch_size = 128
n_epochs = 1000
n_x = 5
n_z = 100
lr = 3e-4

dataset = HyperplaneCachedDataset(25, range(10), n_x)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

dataiter = iter(data_loader)
examples = dataiter.next()[:10]
data = dataset[:]
for i in range(10):
    print data.masked_select(data==i).size()
exit()

print 'Length: %i'%len(dataset)

class VAE(nn.Module):
    def __init__(self, n_x, n_z):
        super(VAE, self).__init__()
        self.n_z = n_z
        self.n_x = n_x

        self.enc1 = nn.Linear(self.n_x, 500)
        self.enc2 = nn.Linear(500, 500)
        self.enc_mu = nn.Linear(500, self.n_z)
        self.enc_logvar = nn.Linear(500, self.n_z)

        self.dec1 = nn.Linear(self.n_z, 500)
        self.dec2 = nn.Linear(500, 500)
        self.dec = nn.Linear(500, 10*self.n_x)

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        z_mu = self.enc_mu(x)
        z_logvar = self.enc_logvar(x)
        return z_mu, z_logvar

    def reparametrize(self, mu, logvar):
        eps = Variable(mu.data.normal_())
        return mu + 0.5*logvar.exp()*eps

    def decode(self, z):
        z = F.relu(self.dec1(z))
        z = F.relu(self.dec2(z))
        x = softmax(self.dec(z).view(-1,self.n_x,10))
        return x

    def forward(self, x):
        z_mu, z_logvar = self.encode(x)
        z = self.reparametrize(z_mu, z_logvar)
        x = self.decode(z)
        return x, z_mu, z_logvar

def criterion(x, x_pred,  z_mu, z_logvar):
    log_px = nll_loss(x, x_pred)
    kl = (0.5*(z_logvar.exp() + z_mu**2 - z_logvar - 1)).sum(-1).mean()
    loss = log_px + kl
    return loss, log_px, kl


vae = VAE(n_x, n_z)
optimizer = optim.Adam(vae.parameters(), lr=lr)

z = Variable(torch.zeros(10,100).normal_())

for e in range(n_epochs):
    for data in data_loader:
        x = Variable(data)
        optimizer.zero_grad()

        x_pred, z_mu, z_logvar = vae(x)
        loss, log_px, kl = criterion(x.long(), x_pred, z_mu, z_logvar)
        loss.backward()
        optimizer.step()

    print loss.data[0], log_px.data[0], kl.data[0]

    _, x_pred  = vae.decode(z).max(2)
    print x_pred.sum(1).data.numpy().squeeze()
    print examples.sum(1).numpy().squeeze()

    x_pred, z_mu, z_logvar = vae(Variable(examples))
    _, x_pred = x_pred.max(2)
    print examples.numpy()
    print x_pred.data.numpy().squeeze()
