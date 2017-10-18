#!/bin/bash
python ../train_mila8.py --datadir ~/data/mila8 --logdir /data/lisa/exp/huanggab/mila8-batch --resolution=$1 --sigma=0.1 --deform-cache 1000 --threads 8 --latent=$2
