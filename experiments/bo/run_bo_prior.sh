#!/bin/bash

NUM_RUNS=$3  # Number of repetitions

# Loop through NUM_RUNS now repetition is set to 1 as we repeat by seed
for SEED in $(seq 0 $((NUM_RUNS-1))); do
    echo "Running iteration with seed=$SEED..."
    
    python -W ignore bo_prior_run.py benchmark=$1 prior_std=$2 benchmark.n_repetition=1 result_path=$4 seed=$SEED 

    sleep 1  # Avoid potential conflicts in launching processes
done