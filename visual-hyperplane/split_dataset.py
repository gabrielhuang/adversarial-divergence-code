import argparse
import cPickloe as pickle
import hyperplane_dataset

parser = argparse.ArgumentParser(description='Split dataset')
# task specific
parser.add_argument('--amount', default=25, type=int, help='target to sum up to')
parser.add_argument('--digits', default=5, type=int, help='how many digits per sequence')
parser.add_argument('--random-seed', default=1234, type=int, help='random seed')
parser.add_argument('-o', '--output', default='combinations.pkl', help='where to save splits')
parser.add_argument('--train-ratio', default=0.8, type=float, help='train size ratio')

args = parser.parse_args()

full, train, test = hyperplane_dataset.get_full_train_test(args.amount, range(10), args.digits, one_hot=False, validation=args.train_ratio, seed=args.random_seed)

dataset = {
        'full': full,
        'train': train,
        'test': test,
        'amount': args.amount,
        'digits': args.digits,
        'train_ratio': args.train_ratio,
}

with open(args.output, 'wb') as fp:
    print 'Writing dataset to {}'.format(args.output)
    pickle.dump(dataset, fp)
