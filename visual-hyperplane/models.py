import torch.nn as nn
import torch.nn.functional as F

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
        x_mu = self.dec_mu(x))
        x_logvar = self.dec_sigma(x))

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
