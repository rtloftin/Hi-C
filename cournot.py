"""Main script for launching experiments from config files."""
import argparse
import itertools
import torch
from torch.multiprocessing import Pool
import traceback

# TODO: Move run and setup scripts to Amanuensis package for reuse - We're not building this anymore
from hi_c import setup_experiments, run_experiment


def print_error(error):  # Callback for python multiprocessing
    traceback.print_exception(type(error), error, error.__traceback__, limit=5)


def parse_args():  # Parse command line arguments - keep this simple
    parser = argparse.ArgumentParser(description=__doc__,  # TODO: Need a docstring
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-o", "--output-path", type=str, default="temp_results/cournot",
                        help="directory in which we should save results")

    # TODO: Check how many steps the trainer does per iteration - wastes time reporting every iteration
    parser.add_argument("--num-iters", type=int, default=1_000,
                        help="the number of experiment iterations")
    parser.add_argument("--steps-per-iter", type=int, default=1_000,
                        help="the number of iterations")

    parser.add_argument("--num-seeds", type=int, default=32,
                        help="the number of random seeds to run, overrides values from the config file")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="a list of random seeds to run, overrides values from the config file")

    parser.add_argument("-n", "--num-cpus", type=int, default=4,
                        help="the number of parallel experiments to launch")
    parser.add_argument("--flush-secs", type=int, default=60,
                        help="number of seconds after which we should flush the training logs (default 60)")

    return parser.parse_args()


def init_configs(args):
    pass


if __name__ == '__main__':
    args = parse_args()  # NOTE: In the `interactive_agents` repo, this is - what, what is it?

    # Select torch device
    device = "cuda" if args.gpu else "cpu"

    # Allow procedurally generated config dictionaries as well
    # Setup experiments  # NOTE: Does this handle "grid_search" commands?
    paths = setup_experiments(args.config_files,
                              args.output_path,
                              num_seeds=args.num_seeds,
                              seeds=args.seeds,
                              arguments=unknown)  # NOTE: In the new framework, experiment directories are initialized for each seed first

    # Limit Torch CPU parallelism - This must be set BEFORE we initialize the process pool
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)

    # Launch experiments
    with Pool(args.num_cpus) as pool:
        experiments = []
        for path in itertools.chain.from_iterable(paths.values()):
            experiments.append(pool.apply_async(run_experiment, (path,), {
                "device": device,
                "flush_secs": args.flush_secs
            },
                                                error_callback=print_error))  # NOTE: The "run_experiment" method just takes the path to the initialized experiment directory

        # Wait for all trails to complete before returning
        for experiment in experiments:
            experiment.wait()
