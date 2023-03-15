import numpy as np
import os
import os.path
import re

if __name__ == '__main__':
    DIR = "results/debug/naive_naive_ipd_2023_03_14_T18_30"

    strategies = []
    regex = re.compile(r"seed_([0-9]*)")
    for obj in os.listdir(DIR):
        match = regex.match(obj)
        if match is not None:
            path = os.path.join(DIR, obj)
            seed = match.group(1)
            if os.path.isdir(path):
                path = os.path.join(path, "artifacts/strategies_a.npy")

                try:
                    strategy = np.load(path)
                    strategies.append((seed, strategy[-1]))
                except:
                    print(f"Warning, could not load strategies for seed {seed}")
    
    for seed, strategy in strategies:
        print(f"{seed} - {strategy}")
