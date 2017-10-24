#!/bin/bash
python ../train_gan_visual.py --logdir /data/lisa/exp/huanggab/hyperplanegan --amount 25 --nb_digits 5 --latent-global $1 --model-generator constrained --model-discriminator constrained --mnist ../data
