import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from sklearn.preprocessing import OneHotEncoder
from coins import generate_combinations
import cPickle as pickle
import numpy as np


class HyperplaneDataset(Dataset):
    '''
    one_hot:
        if true,
            will return a matrix of shape (n_coins, len(coins))
        if false,
            will return a real vector of shape (n_coins)

    Assume coins is a range
    '''
    def __init__(self, combinations, coins, one_hot):
        self.combinations = combinations
        self.coins = coins
        self.one_hot = one_hot
        if self.one_hot:
            len_coins = len(self.coins)
            self.onehot_encoder = OneHotEncoder(sparse=False).fit(np.arange(len_coins).reshape(-1, 1))

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        c = self.combinations[idx]
        if self.one_hot:
            c_encoded = self.onehot_encoder.transform(np.asarray(c).reshape(-1, 1))
            return torch.LongTensor(c_encoded.astype(int))
        else:
            return torch.LongTensor(c)


def generate_hyperplane_dataset(amount, coins, n_coins, one_hot):
        # Generate combinations
        combinations = generate_combinations(amount, coins, n_coins)
        return HyperplaneDataset(combinations, coins, one_hot)


class HyperplaneWithLookup(Dataset):
    def __init__(self, hyperplane_dataset):
        self.hyperplane_dataset = hyperplane_dataset
        self.set = {tuple(c): idx for idx, c in enumerate(self.hyperplane_dataset.combinations)}

    def __len__(self):
        return len(self.hyperplane_dataset)

    def __getitem__(self, idx):
        return self.hyperplane_dataset[idx]

    def __contains__(self, c):
        '''
        for a tuple c = (t1, t2, .., t_n)
        one can test whether c is in the dataset
        '''
        if self.hyperplane_dataset.one_hot:
            c = tuple(c.numpy().argmax(axis=1))
        else:
            c = tuple(c.numpy())
        return c in self.set


class HyperplaneImageDataset(Dataset):
    def __init__(self, hyperplane_dataset, root, train):
        self.hyperplane_dataset = hyperplane_dataset

        self.images = MNIST(root=root, train=train, download=True, transform=ToTensor())
        loader = iter(DataLoader(self.images, batch_size=len(self.images)))
        _, all_labels = loader.next()
        all_labels = all_labels.numpy()
        self.idx = {}
        for i in range(10):
            self.idx[i] = np.argwhere(all_labels==i).squeeze()

    def __len__(self):
        return len(self.hyperplane_dataset)

    def __getitem__(self, idx):
        x = self.hyperplane_dataset[idx]
        image = torch.zeros(len(x),28,28)
        for i in range(len(x)):
            image[i] = self.images[np.random.choice(self.idx[int(x[i])])][0]
        return image

def get_full_train_test(amount, coins, n_coins, one_hot, validation=0.8, seed=None):
    # Get main dataset
    full_dataset = generate_hyperplane_dataset(amount, coins, n_coins, one_hot)

    # Shuffle and split
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(full_dataset.combinations)
    train_end = int(validation * len(full_dataset))
    train_dataset = HyperplaneDataset(full_dataset.combinations[:train_end], coins, one_hot)
    test_dataset = HyperplaneDataset(full_dataset.combinations[train_end:], coins, one_hot)

    # Create lookup tables
    full_lookup = HyperplaneWithLookup(full_dataset)
    train_lookup = HyperplaneWithLookup(train_dataset)
    test_lookup = HyperplaneWithLookup(test_dataset)

    return full_lookup, train_lookup, test_lookup


if __name__ == '__main__':
    # Generate full dataset, train and test splits
    for one_hot in [True, False]:
        print '*'*32
        print 'One hot is true'
        print '*'*32

        full, train, test = get_full_train_test(4, range(4), 3, one_hot=one_hot, seed=0)
        print 'Length Full', len(full)
        print 'Length Train', len(train)
        print 'Length Test', len(test)
        print
        print 'Full', full[:]
        print 'Train', train[:]
        print 'Test', test[:]

        print 'Is training example in training?', train[0] in train
        print 'Is training example in test?', train[0] in test
        print 'Is test example in training?', test[0] in train
        print 'Is test example in test?', test[0] in test

        # Try dataloader
        data_loader = DataLoader(train, batch_size=5, shuffle=True)
        for data in data_loader:
            # This would be the main training loop
            print 'Training batch'
            print
            print "Now here's an example batch"
            print 'You might need to cast to torch.Tensor from torch.LongTensor'
            print data
            break
