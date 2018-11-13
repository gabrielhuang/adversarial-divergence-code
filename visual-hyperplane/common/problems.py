import torch
from torch import nn
import numpy as np
from coins import generate_combinations
from itertools import product
from torch.utils.data import DataLoader

def split(iterable, train_ratio=0.1, shuffle=True):
    if shuffle:
        np.random.shuffle(iterable)
    n = int(len(iterable)*train_ratio)
    train = iterable[:n]
    test = iterable[n:]
    return train, test

def to_onehot(x):
    x_flat = x.flatten()
    x_onehot = np.zeros((len(x_flat), 10), dtype=np.float32)
    x_onehot[range(len(x_onehot)), x_flat] = 1.
    return x_onehot.reshape(x.shape[0], x.shape[1], 10)

CONSTRAINTS = [
    'sum_25',
    'increasing',
    'symmetric',
    'even'
]
ENCODING_MODES = [
    'onehot',
    'embedding',
    'real'    
]
#CONSTRAINT = 'sum_25'
#CONSTRAINT = 'increasing'
#CONSTRAINT = 'symmetric'
CONSTRAINT = 'even'
#ENCODING_MODE = 'onehot'
#ENCODING_MODE = 'embedding'
ENCODING_MODE = 'real'
BATCH = 32

class Problem(object): 
    def __repr__(self):
        return '\n'.join([
         'Problem "{}":[{}]'.format(self.constraint, self.encoding_mode),
         '  train+: {}'.format(len(self.train_positive)),
         '  train-: {}'.format(len(self.train_negative)),
         '  test+: {}'.format(len(self.test_positive)),
         '  test-: {}'.format(len(self.test_negative))])
        
class Summary(object): pass

def make_infinite(iterable):
    while True:
        for i in iterable:
            yield i
            
def get_problem(CONSTRAINT, ENCODING_MODE, train_ratio=0.1):
    p = Problem()

    uniform = np.asarray(list(product(range(10),range(10),range(10),range(10),range(10))))
    if CONSTRAINT == 'sum_25':
        combinations = np.asarray(generate_combinations(25, range(10), 5))
        positive = combinations
        combinations_set = set(map(tuple, combinations))
        non_combinations = np.asarray([c for c in uniform if tuple(c) not in combinations_set])
        negative = non_combinations
    elif CONSTRAINT == 'non_25':
        combinations = np.asarray(generate_combinations(25, range(10), 5))
        negative = combinations
        combinations_set = set(map(tuple, combinations))
        non_combinations = np.asarray([c for c in uniform if tuple(c) not in combinations_set])
        positive = non_combinations
    elif CONSTRAINT == 'uniform':
        positive = uniform
        negative = np.zeros((0, 5), dtype=int)
    elif CONSTRAINT == 'increasing':
        mask = np.all(np.diff(uniform, 1) >= 0, 1)
        positive = uniform[mask]
        negative = uniform[~mask]
    elif CONSTRAINT == 'symmetric':
        mask = np.all(uniform[:,:2] == uniform[:,:2:-1], 1)
        positive = uniform[mask]
        negative = uniform[~mask]
    elif CONSTRAINT == 'even':
        mask = (uniform.sum(1)%2==0)
        positive = uniform[mask]
        negative = uniform[~mask]
    if ENCODING_MODE == 'real':
        positive = positive.astype(np.float32) / 10. # respect range
        negative = negative.astype(np.float32) / 10.
    elif ENCODING_MODE == 'int':
        pass
    else: # positive, negative must be Long before that
        positive = to_onehot(positive)
        negative = to_onehot(negative)

    p.constraint = CONSTRAINT
    p.encoding_mode = ENCODING_MODE
        
    p.positive = positive
    p.negative = negative

    p.train_positive, p.test_positive = split(positive, train_ratio)
    p.train_negative, p.test_negative = split(negative, train_ratio)

    p.train_positive_iter = make_infinite(DataLoader(p.train_positive, batch_size=BATCH, shuffle=True))
    p.test_positive_iter = make_infinite(DataLoader(p.test_positive, batch_size=BATCH, shuffle=True))
    p.train_negative_iter = make_infinite(DataLoader(p.train_negative, batch_size=BATCH, shuffle=True))
    p.test_negative_iter = make_infinite(DataLoader(p.test_negative, batch_size=BATCH, shuffle=True))

        
    return p

