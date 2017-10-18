import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm

def elastic_deform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return map_coordinates(image, indices, order=1).reshape(shape)

class ElasticDeformCached(object):
    def __init__(self, shape, alpha, sigma, cache_size=1000):
        self.cache_size = cache_size
        self.shape = shape
        self.alpha = alpha
        self.sigma = sigma

        self.recompute()

    def deform(self, image):
        i = np.random.randint(len(self.all_indices))
        indices = self.all_indices[i]
        # indices have to be same type as image
        warped = map_coordinates(image, indices, order=1).reshape(self.shape)
        return warped

    def recompute(self):
        self.all_indices = []
        print 'Precomputing deformation filters'
        for i in tqdm(xrange(self.cache_size)):
            dx = gaussian_filter((np.random.rand(*self.shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha
            dy = gaussian_filter((np.random.rand(*self.shape) * 2 - 1), self.sigma, mode="constant", cval=0) * self.alpha

            x, y = np.meshgrid(np.arange(self.shape[0]), np.arange(self.shape[1]))
            new_y = np.reshape(y+dy, (-1, 1))
            new_x = np.reshape(x+dx, (-1, 1))
            self.all_indices.append((new_y, new_x))


