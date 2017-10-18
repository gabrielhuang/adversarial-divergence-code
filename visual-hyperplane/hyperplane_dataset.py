import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder
import coins


class HyperplaneCachedDataset(Dataset):
    def __init__(self, amount, coins, n_coins, one_hot=False, use_float=True):
        '''
        one_hot:
            if true,
                will return a matrix of shape (n_coins, len(coins))
            if false,
                will return a real vector of shape (n_coins)
        '''
        Dataset.__init__(self)
        self.amount = amount
        self.coins = coins
        self.n_coins = n_coins
        self.one_hot = one_hot
        self.use_float = use_float

        self.generate()

        if self.one_hot:
            self.onehot_encoder = OneHotEncoder(sparse=False).fit(np.arange(10).reshape(-1, 1))

    def generate(self):
        self.combinations = coins.aux(self.amount, self.coins, self.n_coins)

    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        c = self.combinations[idx]
        if self.one_hot:
            c_encoded = self.onehot_encoder.transform(np.asarray(c).reshape(-1, 1))
            return torch.Tensor(c_encoded)
        else:
            if self.use_float:
                return torch.Tensor(c)
            else:
                return torch.LongTensor(c)


if __name__ == '__main__':
    dataset = HyperplaneCachedDataset(3, range(10), 3)
    data_loader = DataLoader(dataset, batch_size=5, shuffle=True)
    print 'Length', len(dataset)
    print 'Example', dataset[0]
    print 'Example batch'
    for data in data_loader:
        # This would be the main training loop
        print data
        break

