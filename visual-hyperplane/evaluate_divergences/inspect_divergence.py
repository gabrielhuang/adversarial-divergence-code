#!/usr/bin/env python
import os
import json
import matplotlib.pyplot as plt
import sys
import math
from scipy.ndimage.filters import gaussian_filter1d

#sys.argv += ['runs/run-2018.09.07-15.41.17.DIVERGENCE']

if len(sys.argv) != 2:
    raise ValueError('usage: {} file.DIVERGENCE'.format(sys.argv[0]))

run_dir = os.path.splitext(sys.argv[1])[0]

with open('{}/stats.json'.format(run_dir), 'rb') as fp:
    data = json.load(fp)

print data.keys()

width = int(math.ceil(math.sqrt(len(data))))

smooth_window = 21
for i, (key, value) in enumerate(data.items()):
    # Smooth
    smoothed = gaussian_filter1d(value, smooth_window)

    plt.subplot(width, width, i+1)
    #plt.plot(value)
    plt.plot(smoothed)
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.title(key)

plt.show()
