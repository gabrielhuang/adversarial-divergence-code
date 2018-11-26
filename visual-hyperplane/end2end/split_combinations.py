import argparse
import pickle
import numpy as np

import sys
sys.path.append('..')
from common.problems import get_problem, TASKS, Problem

parser = argparse.ArgumentParser(description='Split dataset')
# task specific
parser.add_argument('--task', default='sum_25', choices=TASKS, help=','.join(TASKS))
parser.add_argument('--criterion', default='sum_25', help='target to sum up to')
parser.add_argument('--random-seed', default=0, type=int, help='random seed')
parser.add_argument('--train-ratio', default=0.5, type=float, help='train size ratio')
parser.add_argument('--output', default='problem.pkl', help='output file')

args = parser.parse_args()

np.random.seed(args.random_seed)

problem = get_problem(args.task, 'int', train_ratio=args.train_ratio)

problem.args = vars(args)

print 'Train positive', problem.train_positive

with open(args.output, 'wb') as fp:
    print 'Writing problem to {}'.format(args.output)
    pickle.dump(problem, fp)
