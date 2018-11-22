import torch
from torch import nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return F.sigmoid(self.fc4(h3))

    def generate(self, batch_size, device='cpu'):
        z = torch.randn(batch_size, 20).to(device)
        x = self.decode(z)
        return x.view(len(x), 1, 28, 28)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class Discriminator1(nn.Module):
    def __init__(self, with_sigmoid):
        nn.Module.__init__(self)

        self.with_sigmoid = with_sigmoid
        layers = [
            nn.Linear(784*5, 400),
            nn.ReLU(),
            nn.Linear(400, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
        ]
        if with_sigmoid:
            layers.append(nn.Sigmoid())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 784*5)  # concatenate all dimensions
        x = self.main(x)
        return x


class Discriminator2(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.shared = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
        )
        self.merge = nn.Sequential(
            nn.Linear(20*5, 80),
            nn.ReLU(),
            nn.Linear(80, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x_per_digit = x.view(len(x)*5, -1)
        embeddings = self.shared(x_per_digit)
        embeddings_merged = embeddings.view(len(x), -1)
        out = self.merge(embeddings_merged)
        return out

    def classify_digit(self, x):
        x = x.view(len(x), -1)  # flatten digits
        x = self.shared(x)
        x = self.classifier(x)
        return x


class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class DiscriminatorCNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        latent_dim = 100
        filters = 64
        self.embedder = nn.Sequential(
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
            # 1
        )

        self.merge = nn.Sequential(
            nn.Linear(latent_dim*5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.get_logits(x)
        out = self.sigmoid(out)
        return out

    def get_logits(self, x):
        x_per_digit = x.view(len(x)*5, 1, 28, 28)
        embeddings = self.embedder(x_per_digit)
        embeddings_merged = embeddings.view(len(x), -1)
        out = self.merge(embeddings_merged)
        return out


class GeneratorCNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.shared_dim = shared_dim = 200
        self.digit_dim = digit_dim = 100
        self.filters = filters = 64
        self.dims = dims = 256

        self.demuxer = nn.Sequential(
            nn.Linear(shared_dim, dims),
            nn.ReLU(),
            nn.BatchNorm1d(dims),
            nn.Linear(dims, dims),
            nn.ReLU(),
            nn.BatchNorm1d(dims),
            nn.Linear(dims, dims),
            nn.ReLU(),
            nn.Linear(dims, 5*digit_dim),
        )

        self.decoder = nn.Sequential(
            ReshapeLayer([-1, digit_dim, 1, 1]),
            # latent:1x1
            nn.ConvTranspose2d(digit_dim, 4*filters, 4, 1, 0),
            nn.ReLU(True),
            nn.BatchNorm2d(4*filters),
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

    def forward(self, noise):
        digit_embeddings = self.demuxer(noise).view(len(noise)*5, -1)
        images = self.decoder(digit_embeddings)
        images_stacked = images.view(len(noise), 5, 28, 28)
        return images_stacked


class Discriminator4(Discriminator2):
    def __init__(self):
        nn.Module.__init__(self)

        self.shared = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
        )
        self.merge = nn.Sequential(
            nn.Linear(5*10, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x_per_digit = x.view(len(x)*5, -1)
        embeddings = self.shared(x_per_digit)
        classified = self.classifier(embeddings)
        classified_merged = classified.view(len(x), -1)
        out = self.merge(classified_merged)
        return out


class Discriminator4exp(Discriminator4):

    def forward(self, x):
        x_per_digit = x.view(len(x)*5, -1)
        embeddings = self.shared(x_per_digit)
        classified = self.classifier(embeddings)
        classified = classified.exp()
        classified_merged = classified.view(len(x), -1)
        out = self.merge(classified_merged)
        return out


class Discriminator4real(Discriminator2):
    def __init__(self):
        nn.Module.__init__(self)

        self.shared = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
        )
        self.merge = nn.Sequential(
            nn.Linear(5*10, 20),
            nn.ReLU(),
            nn.Linear(20, 1),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(20, 1),
        )

    def forward(self, x):
        x_per_digit = x.view(len(x)*5, -1)
        embeddings = self.shared(x_per_digit)
        classified = self.classifier(embeddings)
        classified_merged = classified.view(len(x), -1)
        out = self.merge(classified_merged)
        return out


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

