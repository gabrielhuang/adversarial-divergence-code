#!/bin/bash

ITERATIONS=${1:-200}

python deploy.py --iterations $ITERATIONS --model trained_models/vae_new.torch --use-cuda 1 results-vae
