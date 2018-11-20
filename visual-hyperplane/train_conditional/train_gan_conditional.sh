#!/bin/sh
python train_gan_conditional.py --dataset mnist --digit $1 --dataroot mnist
