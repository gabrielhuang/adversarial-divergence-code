from hyperplane_dataset import HyperplaneCachedDataset
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math, os, argparse

#import visdom
#vis = visdom.Visdom()

def numpy2tuple(x):
    return list(map(tuple,x))

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
        eps = Variable(mu.data.new(mu.size()).normal_())
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

def train(args):
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    lr = args.learning_rate

    optimizer = optim.Adam(vae.parameters(), lr=lr)
    for e in range(n_epochs):
        for i in range(0, len(trainset), batch_size):
            x = Variable(torch.from_numpy(trainset[i:i+batch_size]))
            optimizer.zero_grad()

            x_pred, z_mu, z_logvar = vae(x)
            loss, log_px, kl = criterion(x.long(), x_pred, z_mu, z_logvar)
            loss.backward()
            optimizer.step()
            print 'Epoch: %i, Lowerbound: %.2f, %i/%i'%(e, loss.data[0], i, len(trainset))
        torch.save(vae.state_dict(), os.path.join(output_path, 'checkpoints', '%i.model'%e))

def test(args):
    filename = args.filename
    N = args.num_samples

    vae.load_state_dict(torch.load(filename))

    z = Variable(torch.zeros(N,n_z).normal_())
    _, x_pred  = vae.decode(z).max(2)
    x_pred = x_pred.data.numpy().squeeze()
    #vis.histogram(x_pred.sum(1).squeeze(), win=0, opts=opts)
    x_true = x_pred[x_pred.sum(1).squeeze()==sum_digits]

    x_pred, z_mu, z_logvar = vae(testset)
    _, x_pred = x_pred.max(2)
    #vis.histogram(x_pred.sum(1).data.numpy().squeeze(), win=1, opts=opts)
    print 'L1 error: %.2f, Precision: %.2f, Recall: %.2f'%((testset.float()-x_pred.float()).abs().sum(1).mean().data[0],
     len(x_true)/float(N)*100, len(set(numpy2tuple(x_true)))/float(len(dataset))*100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('-nz', '--latent-dimension', default=10, type=int)
    parser.add_argument('-nx', '--n-digits', default=5, type=int)
    parser.add_argument('-t', '--sum-digits', default=25, type=int)
    parser.add_argument('-ts', '--train-size', default=0.2, type=float)
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('-bs', '--batch-size', default=128, type=int)
    parser_train.add_argument('--n_epochs', default=1000, type=int)
    parser_train.add_argument('-lr', '--learning-rate', default=3e-4, type=float)
    parser_train.set_defaults(func=train)

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('-N', '--num-samples', default=10000, type=int)
    parser_test.add_argument('filename')
    parser_test.set_defaults(func=test)


    args = parser.parse_args()

    n_z = args.latent_dimension
    n_x = args.n_digits
    sum_digits = args.sum_digits
    train_size = args.train_size
    seed = args.seed
    output_path = '/checkpoint/hberard/adversarial_divergence/t_%i_nx_%i_ts_%.2f_nz_%i'%(sum_digits,n_x,train_size,n_z)
    rng = np.random.RandomState(seed=seed)
    torch.manual_seed(seed)

    print 'Loading Dataset...'
    dataset = HyperplaneCachedDataset(sum_digits, range(10), n_x)
    dataset = dataset[:].numpy()
    rng.shuffle(dataset)

    trainset, testset = dataset[:1000], dataset[1000:]
    testset = Variable(torch.from_numpy(testset))

    for i in range(10):
        print '%i appears %i times.'%(i,dataset[dataset==i].size)

    print 'Length: %i'%len(dataset)

    vae = VAE(n_x, n_z)

    if not os.path.exists(os.path.join(output_path, 'checkpoints')):
        os.makedirs(os.path.join(output_path, 'checkpoints'))

    args.func(args)
