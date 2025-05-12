#!/usr/bin/env bash

set -o xtrace

# note that there are N-1 GPUs actually running the search,
# so to evenly divide samples, search samples = batch size * (workers - 1)
srun -l -u python main.py \
    --num-search-samples 12 \
    --load-balance-search \
    --load-balance-batch-size 4 \
    --use-paradigms \
    | tee out.log
