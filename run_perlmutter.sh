#!/usr/bin/env bash

set -o xtrace

# note that there are N-1 GPUs actually running the search,
# so to evenly divide samples, search samples = batch size * (workers - 1)
srun -l -u python main.py \
    --num-inference-steps 100 \
    --image-decode-step 5 \
    --num-search-samples 12 \
    --search-batch-size 4 \
    --load-balance-search \
    --num-search-inference-steps 100 \
    --use-paradigms \
    --early-stop-dynamic variance \
    --early-stop-dynamic-threshold 0.1 \
    --early-stop-dynamic-window 5 \
    --early-stop-dynamic-timestep-start 700 \
    --save-intermediate-images \
    --verbose \
    --disable-progress-bars \
    | tee out.log

# srun -l -u python main.py \
#     --num-search-samples 12 \
#     --load-balance-search \
#     --load-balance-batch-size 4 \
#     --use-paradigms \
#     | tee out.log
