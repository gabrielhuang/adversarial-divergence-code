# Visual Hyperplane Experiments

This experiment compares a Nonparametric divergence (KL/Maximum Likelihood)
and a Parametric divergence (Regularized GAN) for enforcing toy constraints 
(MNIST digits have to sum up to 25).

There are two steps:
- Analyzing divergence with fixed distributions.
- End-to-end generation of MNIST digits summing up to 25.

## Analyzing Divergence with Fixed Distribution

### 1. Train Conditional Visual Models.

```cd train_conditional/```

Variational AutoEncoder, where `X=0,1,...,9`:

```python train_vae_conditional.py --digit X```

Generative Aversarial Network:

```python train_gan_conditional.py --digit X --dataset mnist --dataroot mnist```
or
```./train_gan_conditional.sh X```


### 2. Evaluate if Nonparametric divergence (KL/ML) can enforce Sum-25.

```cd evaluate_divergences/```

Then run the jupyter notebook `evaluate_joint_likelihood.ipynb`

*Main result*:

Likelihood(Test-25||VaeEpoch80-Non25) - Likelihood(Test-25||VaeEpoch70-Non25) >> 
Likelihood(Test-25||VaeEpoch70-25) - Likelihood(Test-25||VaeEpoch70-Non25)

In other words, Likelihood focuses mostly on visual appearance, and there is almost no gain for enforcing Sum-25 constraint.

-> same experiment for GAN?


### 3. Evaluate if Parametric divergence (Regularized GAN) can enforce Sum-25.

```cd evaluate_divergences/```

We know from 4. that discriminator can successfully enforce and generalize Sum-25 constraint.
Would it be able to do it in a GAN setting, when generated digits are still imperfect and don't sum up to 25?


Train a discriminator on Test-25/Vae-Non25. 
It is possible to distinguish those distributions solely based on the images (in fact only a single image probably suffices, so there is no need to learn the joint probability of the images).

Evaluate accuracy on telling apart:
- Test-25/Test-Non25 to see if discriminator did learn Sum-25 constraint. The discriminator is barely able to distinguish them.

*Main result*:

When trained on Test-25/Vae-Non25, which simulates a GAN setting (generated digits imperfect and don't sum to 25),
the discriminator tends to fixate on visual properties (because they allow 100% accuracy), and will not enforce the symbolic constraints.

Todo:
- try with Test-25/GAN-Non25. It might have a better chance of working since the digits are better.


### 4. Evaluate if Parametric divergence (Regularized GAN) can enforce Sum-25.


We divide the ~6000 combinations that sum up to 25 into two disjoints sets 25A and 25B.
We divide the remaining ~100000 combinations that do not sum up to 25 into two disjoint sets 25A and 25B.

Train a discriminator on Test-25A/Test-Non25A.
Evaluate accuracy on telling apart:
- Test-25B/Test-Non25B to see if discriminator can generalize over combinations.
- Test-25A/Vae-Non25A and Vae-25A/Test-Non25A  to see if discriminator can be discriminative to imperfect images.
- Test-25B/Vae-Non25B and Vae-25B/Test-Non25B combines the two previous tasks: discriminator must tell whether imperfect images satisfy a constraint (after generalizing).

```
cd evaluate_divergences/
python evaluate_divergences.py  # train discriminator
python divergence_gui.py  # plot results
```
