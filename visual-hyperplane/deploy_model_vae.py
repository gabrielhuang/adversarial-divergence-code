import json
import os
from models_gan import MNISTNet
from models import VAE
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--iterations', default=70, type=int)
parser.add_argument('--batch-size', default=1000, type=int)
parser.add_argument('--use-cuda', default=1, type=int)
parser.add_argument('--model', required=True, help='path to generator.torch')
parser.add_argument('--digits', default=5, help='number of digits')
parser.add_argument('--classifier', default='trained_models/mnist_classifier.torch', help='number of latent dimensions')
parser.add_argument('output', help='output dir, usually one of: results-vae|results-gan')

args = parser.parse_args()

if not os.path.exists(args.output):
    print 'Creating path {}'.format(args.output)
    os.makedirs(args.output)

netG = torch.load(args.model, map_location=lambda storage, loc: storage)
netG.eval()

classifier = torch.load(args.classifier, map_location=lambda storage, loc: storage)
classifier.eval()

digits_list = []
softmax_list = []

if args.use_cuda:
    netG.cuda()

for i in tqdm(xrange(args.iterations)):
    samples = netG.generate(args.batch_size, use_cuda=args.use_cuda).cpu()
    size = samples.size()
    samples = samples.view(-1, 1, size[-2], size[-1])
    logits = classifier(samples)
    probs = F.softmax(logits)
    softmax, digits = probs.data.max(1)
    digits = digits.view(-1, args.digits)
    softmax = softmax.view(-1, args.digits)
    softmax_list = softmax_list + list(softmax.numpy())
    digits_list = digits_list + list(digits.numpy())

print 'saving data ...'
np.save('{}/digits.npy'.format(args.output), digits_list)
np.save('{}/softmax.npy'.format(args.output), softmax_list)
