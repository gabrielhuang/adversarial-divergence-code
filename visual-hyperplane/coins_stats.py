import numpy as np
from coins import generate_combinations
np.set_printoptions(precision=4)

n_coins = 5
combinations = np.asarray(generate_combinations(25, range(10), n_coins))
print 'Total combinations', combinations.shape

# Get marginals
marginals = []
for i in xrange(10):
    marginals.append((combinations == i).mean())
print 'Marginal distributions', np.asarray(marginals)

H_indie = np.sum(-m * np.log(m) for m in marginals)
print 'Independent entropy', H_indie

# Get joint entropy. It is a uniform over the number of combinations
H_joint = np.log(len(combinations))
print 'Joint entropy', H_joint

H_uniform = 5*np.log(10)
print 'Entropy for uniform distribution', H_uniform


# Comparison with 28 x 28
sigma = 0.1
error = 0.1
dims = 28*28
image_likelihood = 0.5 * n_coins * dims * error / sigma

# If likelihood was normalized, then it would make image *very* blurry.
val_mse = 860
val_kl = 29
val_loss = val_mse + val_kl
print 'Typical size of image-nll term', val_mse
