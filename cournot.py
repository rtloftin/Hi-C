"""Runs experiments in the Cournot competition game."""
import argparse
from copy import deepcopy
import itertools
import math
import torch
from torch.multiprocessing import Pool
import traceback

from hi_c import setup_experiments, run_experiment

BASE_CONFIG = {
    "iterations": 1000,
    "trainer": "default",
    "config": {
        "steps_per_iter": 1000,
        "game": "cournot",
        "game_config": {
            "initial_price": 50.,
            "price_slope": 1.,
            "cost_1": 1.,
            "cost_2": 1.,
        }
    }
}

HI_C_CONFIG = {
    "name": "hi_c",
    "params": {
        "lr": {
            "p_series": {
                "scale": 0.001,
                "exponent": 0.1
            }
        },
        "p": {
            "p_series": {
                "scale": 1.,
                "exponent": 0.6
            }
        },
        "baseline_lambda": 0.9,
        "burn_in": 50
    }
}

NAIVE_CONFIG = {
    "name": "naive",
    "params": {"lr": 0.1}
}


def init_configs():

    # Compute appropriate commitment schedule for Hi-C
    initial_price = BASE_CONFIG["config"]["game_config"]["initial_price"]
    price_slope = BASE_CONFIG["config"]["game_config"]["price_slope"]
    perturbation_exponent = HI_C_CONFIG["params"]["lr"]["p_series"]["exponent"]
    inner_lr = NAIVE_CONFIG["params"]["lr"]

    B = 2. * initial_price
    z = math.log(1. - inner_lr * 2. * price_slope)
    commitment_scale = -2. * (perturbation_exponent + 1.) / z
    commitment_offset = -2. * math.log(B) / z

    HI_C_CONFIG["params"]["k"] = {
        "logarithmic": {
            "scale": commitment_scale,
            "offset": commitment_offset
        }
    }

    # Initialize experiment configs
    configs = {
        "hi_c_naive_cournot": deepcopy(BASE_CONFIG),
        "naive_naive_cournot": deepcopy(BASE_CONFIG)
    }

    configs["hi_c_naive_cournot"]["config"]["learners"] = [
        HI_C_CONFIG,
        NAIVE_CONFIG
    ]

    configs["naive_naive_cournot"]["config"]["learners"] = [
        NAIVE_CONFIG,
        NAIVE_CONFIG
    ]

    return configs


def print_error(error):  # Callback for python multiprocessing
    traceback.print_exception(type(error), error, error.__traceback__, limit=5)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-o", "--output-path", type=str, default="temp_results/cournot",
                        help="directory in which we should save results")

    parser.add_argument("--num-seeds", type=int, default=32,
                        help="the number of random seeds to run, overrides values from the config file")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="a list of random seeds to run, overrides values from the config file")

    parser.add_argument("-n", "--num-cpus", type=int, default=4,
                        help="the number of parallel experiments to launch")
    parser.add_argument("--flush-secs", type=int, default=60,
                        help="number of seconds after which we should flush the training logs (default 60)")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Setup experiments
    paths = setup_experiments(init_configs(),
                              args.output_path,
                              num_seeds=args.num_seeds,
                              seeds=args.seeds)

    # Limit Torch CPU parallelism - This must be set BEFORE we initialize the process pool
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)

    # Launch experiments
    with Pool(args.num_cpus) as pool:
        experiments = []
        for path in itertools.chain.from_iterable(paths.values()):
            experiments.append(pool.apply_async(run_experiment, (path,), {
                "device": "cpu",
                "flush_secs": args.flush_secs
            }, error_callback=print_error))

        # Wait for all trails to complete before returning
        for experiment in experiments:
            experiment.wait()
