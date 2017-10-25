
import json
from models_gan import ImageGenerator, UnconstrainedImageGenerator, MNISTNet
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

nb_epochs = 70
batch_size = 1000
use_cuda = 0


class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)


models_dir = 'trained_models'
# gan model
generator_filename = '{}/generator_99000.torch'.format(models_dir)
args_filename = '{}/args.json'.format(models_dir)
# mnist classifier
mnist_filename = '{}/mnist_classifier.torch'.format(models_dir)

with open(args_filename, 'rb') as f:
    args_dict = json.load(f)
args = Bunch(args_dict)


if args.model_generator == 'unconstrained':
    netG = UnconstrainedImageGenerator(args.latent_global, args.nb_digits, args.unconstrained_size)
else:
    netG = ImageGenerator(args.latent_global, args.latent_local, args.nb_digits)

netG = torch.load(generator_filename, map_location=lambda storage, loc: storage)
netG.eval()

classifier = MNISTNet()
classifier = torch.load(mnist_filename, map_location=lambda storage, loc: storage)
classifier.eval()

digits_list = []
softmax_list = []

for i in xrange(nb_epochs):
    samples = netG.generate(batch_size, use_cuda=use_cuda)
    size = samples.size()
    samples = samples.view(-1, 1, size[-2], size[-1])
    logits = classifier(samples)
    probs = F.softmax(logits)
    softmax, digits = probs.data.max(1, keepdim=True)
    digits = digits.view(-1, args.nb_digits)
    softmax = softmax.view(-1, args.nb_digits)
    softmax_list = softmax_list + list(softmax.numpy())
    digits_list = digits_list + list(digits.numpy())

print 'saving data ...'
np.save('{}/gan_digits.npy'.format(models_dir), digits_list)
np.save('{}/gan_softmax.npy'.format(models_dir), softmax_list)


# sum_digits = digits.sum(1).numpy()
# print 'precision: ', np.sum(sum_digits == args.amount)/float(nb_samples) * 100
# plt.hist(sum_digits)
# plt.show()
# plot for sanity check
# gallery = torchvision.utils.make_grid(samples.data, nrow=args.nb_digits, normalize=True, range=(0, 1))
# plt.imshow(np.transpose(gallery.numpy(), (1, 2, 0)), interpolation='nearest')
# plt.show()



