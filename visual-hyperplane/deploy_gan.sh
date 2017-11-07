#!/bin/bash

ITERATIONS=${1:-70}

python deploy.py --iterations $ITERATIONS --model trained_models/generator_99000.torch --use-cuda 1 results-gan
