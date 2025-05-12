#!/usr/bin/env bash

set -o xtrace

salloc -C gpu -N 1 --ntasks-per-node=4 --gpus-per-node=4 -t 00:30:00 -q interactive -A mp309
# salloc -C gpu --ntasks-per-node=1 --gpus-per-node=1 -t 00:10:00 -q interactive -A mp309
