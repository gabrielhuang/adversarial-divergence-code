#!/bin/bash

ITERATIONS=${1:-200}

python count.py --iterations $ITERATIONS --model trained_models/vae-99000.torch --batch-size 1000 results_vae
