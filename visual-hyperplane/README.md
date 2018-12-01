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

#### Main result: Maximum Likelihood fixates on visual appearance.

Likelihood(Test-25||VaeEpoch80-Non25) - Likelihood(Test-25||VaeEpoch70-Non25) >> 
Likelihood(Test-25||VaeEpoch70-25) - Likelihood(Test-25||VaeEpoch70-Non25)

In other words, Likelihood focuses almost exclusively on visual appearance. There is almost no gain for enforcing Sum-25 constraint. In constrast, slightly improving the visual model (epoch 70->80) improves the likelihood significantly.

#### Same experiment on GAN?
It is *much harder* to do same experiment for GAN: divergence values are not stable and depend on network initialization and sampling of the data during SGD. Some formulations (WGAN/WGAN-GP) are more stable than others (Unregularized GAN), but in general the values are hard to compare.

Besides, if the distributions are visually very different (e.g. Test-25/Vae-25), the discrimininator will converge to 100% accuracy. Of course we could prevent it of having 100% accuracy by regularizing, but the point is that the value is hard to interpret in general.


### 3. Can Discriminator detect Sum-25 constraint with imperfect samples?

```cd evaluate_divergences/```

We know from 4. that discriminator can successfully enforce and generalize Sum-25 constraint.
Would it be able to do it in a GAN setting, when generated digits are still imperfect and don't sum up to 25?


Train a discriminator on Test-25/Vae-Non25. 
It is possible to distinguish those distributions solely based on the images (in fact only a single image probably suffices, so there is no need to learn the joint probability of the images).

Evaluate accuracy on telling apart:
- Test-25/Test-Non25 to see if discriminator did learn Sum-25 constraint. The discriminator is barely able to distinguish them.

#### Main result: In the GAN setting the discriminator will not enforce the constraint if generated samples are not good enough.

When trained on Test-25/Vae-Non25, which simulates a GAN setting (generated digits imperfect and don't sum to 25),
the discriminator tends to fixate on visual properties (because they allow 100% accuracy), and will not enforce the symbolic constraints.

#### Todo
- try with Test-25/GAN-Non25. It might have a better chance of working since the digits are better.

#### Extrapolation
This might explain why samples from GANs seem to lack global consistency as scenes. Since images are already imperfect, the discriminator can tell them apart without need to enforce any physical constraints.


### 4. Using a side-task to make discriminator enforce Sum-25 constraint.

#### Motivation: generate digits that sum up to 25.

We introduce a side task (Test-25A/Test-Non25A) to force the discriminator to learn the constraint.

#### Experiment
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

#### Main result: we can make discriminator enforce the constraint using a side task.

This shows:
- Test-25B/Test-Non25B: discriminator can generate new combinations and not just overfit on training combinations (good for **sample complexity**)
- Test-25A/Vae-Non25A and Vae-25A/Test-Non25A: discriminator can enforce constraint even with distribution shift (suggests we can combine side-task with GAN).

## End to end training

### 1. Train models

```
cd end2end/
python train_vae_visual.py
python train_gan_visual.py --side-task 1
python train_gan_visual.py --side-task 0
```

### 2. Analyze results and plot

```
cd analyze/
python count.py --model trained_models/wgan_side.torch --batch-size 100 results_wgan_side
python count.py --model trained_models/wgan_noside.torch --batch-size 100 results_wgan_noside
python count.py --model trained_models/vae-99000.torch --batch-size 100 results_vae
# creates folders analyze/results_wgan_side, ...

python plot_barchart.py
# get plots in analyze/plots
```


#### Todo:
- A/B separation DONE
- Vae-25/Test-Non25 DONE (flipped)
- Same with GANs DONE
