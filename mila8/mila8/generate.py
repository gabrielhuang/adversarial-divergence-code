import sys
import os
from PIL import Image
from scipy.misc import imsave
import numpy as np

if len(sys.argv) != 4:
	print 'args', sys.argv
	print 'usage: {} foldername number resolution'.format(sys.argv[0])
	sys.exit(-1)

foldername = sys.argv[1]
n = int(sys.argv[2])
w = int(sys.argv[3])

os.makedirs(foldername)

with open('{}/draw.bat'.format(foldername), 'w') as fp:
	for i in xrange(n):
		fname = '{}/img_{}.png'.format(foldername, i)
		content = 255*np.ones((w, w), dtype=np.uint8)
		imsave(fname, content)
		fp.write('mspaint img_{}.png\n'.format(i))

