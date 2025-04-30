#!/usr/bin/env bash

salloc -C gpu --ntasks-per-node=4 --gpus-per-node=4 -t 00:30:00 -q interactive -A mp309
