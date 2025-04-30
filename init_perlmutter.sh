#!/usr/bin/env bash

module load python/3.12
module load gpu
module load pytorch
export HF_HOME="$SCRATCH/.cache/huggingface/"

# pip install -r requirements.txt --extra-index-url=https://download.pytorch.org/whl/cu124
