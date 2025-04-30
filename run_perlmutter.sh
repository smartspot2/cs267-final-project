#!/usr/bin/env bash

srun -l -u python main.py | tee out.log
