from hyperplane_dataset import HyperplaneCachedDataset
from torch.utils.data import DataLoader

def softmax(self, input):
    output = input._zeros()
    for i in range():


batch_size = 128
n_epochs = 1000
n_x = 5
n_z = 100
lr = 3e-4

dataset = HyperplaneCachedDataset(25, range(10), n_x)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print 'Length: %i'%len(dataset)

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

class VAE(nn.Module):
    def __init__(self, n_x, n_z):
        super(VAE, self).__init__()
        self.n_z = n_z
        self.n_x = n_x

        self.enc1 = nn.Linear(10*self.n_x, 500)
        self.enc2 = nn.Linear(500, 500)
        self.enc_mu = nn.Linear(500, self.n_z)
        self.enc_logvar = nn.Linear(500, self.n_z)

        self.dec1 = nn.Linear(self.n_z, 500)
        self.dec2 = nn.Linear(500, 500)
        self.dec = nn.Linear(500, 10*self.n_x)

    def encode(self, x):
        x = F.relu(self.enc1(x.view(-1, 10*n_x)))
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
        x = F.softmax(self.dec(z.view(-1,10))).view(-1,)
        return x

    def forward(self, x):
        z_mu, z_logvar = self.encode(x)
        z = self.reparametrize(z_mu, z_logvar)
        x_mu, x_logvar = self.decode(z)
        return x_mu, x_logvar, z_mu, z_logvar

def criterion(x, x_mu, x_logvar, z_mu, z_logvar, eps=1e-6):
    log_px = (-0.5*(math.log(2*math.pi) + x_logvar + (x-x_mu)**2/(x_logvar.exp() + eps))).sum(-1).mean()
    kl = (0.5*(z_logvar.exp() + z_mu**2 - z_logvar - 1)).sum(-1).mean()
    loss = -log_px + kl
    return loss, log_px, kl


vae = VAE(n_x, n_z)
optimizer = optim.Adam(vae.parameters(), lr=lr)

for e in range(n_epochs):
    for data in data_loader:
        x = Variable(data)
        optimizer.zero_grad()

        x_mu, x_logvar, z_mu, z_logvar = vae(x)
        loss, log_px, kl = criterion(x, x_mu, x_logvar, z_mu, z_logvar)
        loss.backward()
        optimizer.step()
        print loss.data[0]
