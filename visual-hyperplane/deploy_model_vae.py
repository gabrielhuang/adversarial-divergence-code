import json
from models_gan import MNISTNet
from models import VAE
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

nb_epochs = 5
#nb_epochs = 70
batch_size = 1000
use_cuda = 1

models_dir = 'trained_models/'
output_dir = 'results'
epoch = 99000
ndigits = 5
nlatent = 200
amount = 25

# gan model
generator_filename = '%s/vae-%i.torch'%(models_dir,epoch)
# mnist classifier
mnist_filename = 'trained_models/mnist_classifier.torch'

netG = VAE(ndigits, nlatent)

netG = torch.load(generator_filename, map_location=lambda storage, loc: storage)
netG.eval()

classifier = MNISTNet()
classifier = torch.load(mnist_filename, map_location=lambda storage, loc: storage)
classifier.eval()

digits_list = []
softmax_list = []

if use_cuda:
    netG.cuda()

for i in tqdm(xrange(nb_epochs)):
    samples = netG.generate(batch_size, use_cuda=use_cuda).cpu()
    size = samples.size()
    samples = samples.view(-1, 1, size[-2], size[-1])
    logits = classifier(samples)
    probs = F.softmax(logits)
    softmax, digits = probs.data.max(1)
    digits = digits.view(-1, ndigits)
    softmax = softmax.view(-1, ndigits)
    softmax_list = softmax_list + list(softmax.numpy())
    digits_list = digits_list + list(digits.numpy())

print 'saving data ...'
np.save('{}/vae_digits.npy'.format(output_dir), digits_list)
np.save('{}/vae_softmax.npy'.format(output_dir), softmax_list)


sum_digits = digits.sum(1).numpy()
print digits.size()
nb_samples = len(digits)
print 'precision: ', np.sum(sum_digits == amount)/float(nb_samples) * 100
plt.hist(sum_digits, bins=5*9, range=(0,5*9))
plt.savefig('%s/hist.png'%output_dir)
# plot for sanity check
#gallery = torchvision.utils.make_grid(samples.data, nrow=ndigits, normalize=True, range=(0, 1))
samples = samples.view(-1,ndigits,1,28,28)[:10]
samples = samples.view(-1,1,28,28)
torchvision.utils.save_image(samples.data, '%s/images.pdf'%output_dir, nrow=ndigits, normalize=True, range=(0,1))
print digits[:10].numpy()
