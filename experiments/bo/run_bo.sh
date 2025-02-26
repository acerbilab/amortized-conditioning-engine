#!/bin/bash

# Parameters (Set manually)
NUM_RUNS=$2  # Number of repetitions

# Loop through NUM_RUNS instead of using SLURM array
for SEED in $(seq 0 $((NUM_RUNS-1))); do
    echo "Running iteration with seed=$SEED..."
    
    python -W ignore bo_run.py benchmark=$1 benchmark.n_repetition=1 benchmark.iters=50 result_path=$3 seed=$SEED 

    sleep 1  # Avoid potential conflicts in launching processes
done


#sbatch batch_bo.sh 4d_ackley $n_rep $ace_4d_model_path $result_path_4d