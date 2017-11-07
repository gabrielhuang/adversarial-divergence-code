#!/bin/bash

ITERATIONS=${1:-70}

python deploy.py --iterations $ITERATIONS --model trained_models/vae-99000.torch --use-cuda 1 results-vae
