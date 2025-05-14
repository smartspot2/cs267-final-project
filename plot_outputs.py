import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("folders", type=str, nargs="+")
parser.add_argument("--gpus", type=int, required=True)

args = parser.parse_args()
fig, axes = plt.subplots(ncols=len(args.folders))

for idx, folder in enumerate(args.folders):
    if len(args.folders) > 1:
        ax = axes[idx]
    else:
        ax = axes

    ax.set_title(folder)
    base_folder = folder.rstrip("/")
    search_folder = f"{base_folder}/search"

    for gpu in range(1, args.gpus):
        gpu_folder = f"{search_folder}/{gpu}"

        for file in os.listdir(gpu_folder):
            basename, ext = os.path.splitext(file)
            if ext != ".json":
                continue

            file_path = os.path.join(gpu_folder, file)
            print(file_path)

            with open(file_path, "r", encoding="utf-8") as f:
                parsed = json.load(f)

            scores = parsed["scores"]

            score_arr = np.array(list(scores.values()))
            timestep_arr = np.array(list(map(int, scores.keys())))
            print(score_arr)

            ax.plot(timestep_arr, score_arr)

    ax.set_xlim(xmin=0, xmax=1000)
    ax.xaxis.set_inverted(True)
plt.show()
