# Visual Hyperplane Experiments

This experiment compares KL-divergence (nonparametric) 
and a GAN-divergence for enforcing toy constraints 
(MNIST digits have to sum up to 25).

There are two steps:
- Analyzing divergence with fixed distributions.
- End-to-end generation of MNIST digits summing up to 25.

## Analyzing Divergence with Fixed Distribution

### 1. Train Conditional Visual Models.

```cd train_conditional/```

Variational AutoEncoder, where `X=0,1,...,9`:

```python train_vae_visual.py --digit X```

Generative Aversarial Network:

```python train_gan_conditional.py --digit X```


### 2. Evaluate Likelihood



### 3. Evaluate Parametric Divergence.

We divide the ~6000 combinations that sum up to 25 into two disjoints sets 25A and 25B.
We divide the remaining ~100000 combinations that do not sum up to 25 into two disjoint sets 25A and 25B.

Train a discriminator on Test-25A/Test-Non25A.
Evaluate on:
- Test-25B/Test-Non25B to see if discriminator can generalize over combinations.
- Test-25A/Vae-Non25A and Vae-25A/Test-Non25A  to see if discriminator can be discriminative to imperfect images.
- Test-25B/Vae-Non25B and Vae-25B/Test-Non25B combines the two previous tasks: discriminator must tell whether imperfect images satisfy a constraint (after generalizing).

```
cd evaluate_divergences/
python evaluate_divergences.py
```
