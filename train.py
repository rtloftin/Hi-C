import argparse
import numpy as np
import torch
from torch.multiprocessing import Pool
import traceback

def print_error(error):
    traceback.print_exception(type(error), error, error.__traceback__, limit=5)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,  # NOTE: Need a docstring
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    # NOTE: Add support for launching from an existing trial directory
    parser.add_argument("config_files", type=str, nargs="*",
                        help="provide one or more experiment config files")
    parser.add_argument("-o", "--output-path", type=str, default="results/debug",
                        help="directory in which we should save results")

    parser.add_argument("--num-seeds", type=int,
                        help="the number of random seeds to run, overrides values from the config file")
    parser.add_argument("--seeds", type=int, nargs="+",
                        help="a list of random seeds to run, overrides values from the config file")

    parser.add_argument("-n", "--num-cpus", type=int, default=1,
                        help="the number of parallel experiments to launch")
    parser.add_argument("-g", "--gpu", action="store_true",
                        help="enable GPU acceleration if available")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print data for every training iteration")
    parser.add_argument("--flush-secs", type=int, default=200,
                        help="number of seconds after which we should flush the training longs (default 200)")
    
    parser.add_argument("-c", "--continue", type=str, default="results/debug",
                        help="directory in which we should save results")

    return parser.parse_args()

## NEW MAIN

if __name__ == '__main__':
    args, unknown = parse_args()

    # NOTE: seed overrides and config loading could be managed within the setup function

    # Load configuration files
    experiments = load_configs(args.config_files)

    # Select torch device  # NOTE: How would this handle multi-GPU machines?
    device = "cuda" if args.gpu else "cpu"
    print(f"Training with Torch device '{device}'")

    
    # Override config if random seeds are provided
    for name, config in experiments.items():
        if args.num_seeds is not None:
            config["num_seeds"] = args.num_seeds

        if args.seeds is not None:
            config["seeds"] = args.seeds
            
        # Add custom arguments to config
        config["arguments"] = unknown

    # Setup experiment
    trial_configs = setup_experiments(experiments, args.output_path, use_existing=False)  # NOTE: What does this return?

    # Limit Torch CPU parallelism - This must be set BEFORE we initialize the process pool
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)

    # Launch experiments
    with Pool(args.num_cpus) as pool:
        trials = []
        for trial in trial_configs:
            trials.append(pool.apply_async(run_trial, (trial,), {
                    "device": device, 
                    "verbose": args.verbose, 
                    "flush_secs": args.flush_secs
                }, error_callback=print_error))
            
        # Wait for all trails to complete before returning
        for trial in trials:
            trial.wait()


## OLD MAIN

if __name__ == '__main__':
    discount = 0.96
    game = IteratedGame(POTENTIAL_GAME, discount)
    # game = TandemGame()

    lr = 0.001
    learner_a = NaiveLearner(game, lr)
    # learner_a = LOLA(game, lr, lr)
    # learner_a = HierarchicalGradient(game, lr * 0.1)
    # learner_a = HiC(game, lr)
    learner_b = NaiveLearner(ReverseGame(game), lr)

    strategy_a, strategy_b = learn(game, learner_a, learner_b, 100)

    print(f"Player 1's final strategy: {strategy_a}")
    print(f"Player 2's final strategy: {strategy_b}")

    exit()

    matrix = torch.tensor([[1, 0, 0, 1]], requires_grad=True, dtype=torch.float)
    inverse = torch.linalg.inv(torch.reshape(matrix, (2, 2)))
    sum = torch.sum(inverse)

    grad,  = torch.autograd.grad([sum], [matrix], create_graph=True)

    print(grad)

    hessian = torch.autograd.grad([torch.sum(grad)], [matrix])

    print(hessian)

    exit()

    A = torch.tensor(2, requires_grad=True, dtype=torch.float)
    B = torch.tensor(3, requires_grad=True, dtype=torch.float)

    output = (A**2) + (B**2)
    gradient_A, gradient_B = torch.autograd.grad([output], [A, B], create_graph=True)

    print(f"output: {type(output)}")
    print(f"gradient w.r.t. A: {gradient_A}")
    print(f"gradient w.r.t. B: {gradient_B}")

    hessian_A_A, hessian_A_B = torch.autograd.grad([gradient_A], [A, B], allow_unused=True)

    print(f"hessian A-B: {hessian_A_A}")
    print(f"hessian A-B: {hessian_A_B}")
