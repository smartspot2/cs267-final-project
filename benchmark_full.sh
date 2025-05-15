#!/usr/bin/env bash

mkdir -p "$SCRATCH/logs"

for seed in 1000 2000; do
    for nodes in 4 3 2 1; do
        ngpu=$((nodes * 4))
        for samples in 4 8 12 16; do
            total_samples=$((ngpu * samples))
            printf "\n===== %s nodes, %s gpus, %s samples per gpu, %s total samples, seed %s =====\n" "$nodes" "$ngpu" "$samples" "$total_samples" "$seed"
            srun -N $nodes --gpus-per-node 4 --ntasks-per-node 4 \
                -l -u python main.py \
                --num-inference-steps 100 \
                --num-search-samples $total_samples \
                --search-batch-size 4 \
                --num-search-inference-steps 100 \
                --use-paradigms \
                --seed $seed \
                --verbose \
                --disable-progress-bars |
                tee "$SCRATCH/logs/full_search_ngpu${ngpu}_totsamps${total_samples}_seed${seed}.log"
        done
    done
done

# one node, 2 gpus
# for seed in 1000 2000; do
#     for samples in 4 8 12 16; do
#         total_samples=$((2 * samples))
#         printf "\n===== 1 node, 2 gpus, %s samples per gpu, %s total samples, seed %s =====\n" "$samples" "$total_samples" "$seed"
#         srun -N 1 --gpus-per-node 2 --ntasks-per-node 2 \
#             -l -u python main.py \
#             --num-inference-steps 100 \
#             --image-decode-step 5 \
#             --num-search-samples $total_samples \
#             --search-batch-size 4 \
#             --load-balance-search \
#             --num-search-inference-steps 100 \
#             --use-paradigms \
#             --early-stop-dynamic variance \
#             --early-stop-dynamic-threshold 0.1 \
#             --early-stop-dynamic-window 5 \
#             --early-stop-dynamic-timestep-start 700 \
#             --seed $seed \
#             --verbose \
#             --disable-progress-bars |
#             tee "$SCRATCH/logs/early_stop_dynamic_ngpu2_totsamps${total_samples}_seed${seed}.log"
#     done
# done
#
#
