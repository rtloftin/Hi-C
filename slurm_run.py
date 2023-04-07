"""DO NOT RUN THIS SCRIPT MANUALLY

This script is called by the `sbatch` script created when
running `train_slurm.py`. 
"""

import argparse
import torch

from hi_c import run_experiment

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # TODO: Add support for launching from an existing trial directory
    parser.add_argument("path", type=str,
                        help="path to initialized experiment directory")

    # NOTE: Currently we don't support multiple CPUs per trial
    # parser.add_argument("-n", "--num-cpus", type=int, default=1,
    #                     help="the number of parallel experiments to launch")
    
    parser.add_argument("-g", "--gpu", action="store_true",
                        help="enable GPU acceleration if available")
    
    parser.add_argument("--flush-secs", type=int, default=200,
                        help="number of seconds after which we should flush the training longs (default 60)")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Select torch device
    device = "cuda" if args.gpu else "cpu"

    # Limit Torch CPU parallelism
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)

    # Launch experiment
    run_experiment(args.path, 
                   device=device, 
                   flush_secs=args.flus_secs)