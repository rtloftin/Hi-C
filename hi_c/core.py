from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import numpy as np
import os
import os.path
import pandas
from tensorboardX import SummaryWriter
import yaml

from hi_c.grid_search import grid_search
from hi_c.experiment import get_experiment_class


def get_timestamp():
    return datetime.now(timezone.utc).strftime("%Y_%m_%d_T%H_%M")


def get_experiment_dir(base_path, name, index_digits=3):
    name = name + "_" + get_timestamp()
    path = os.path.join(base_path, name)

    idx = 0
    while os.path.exists(path):
        idx += 1
        path = os.path.join(base_path, name + "_" + str(idx).zfill(index_digits))
    
    os.makedirs(path)
    return path


def setup_seed(base_path, name, config, seed):
    config = deepcopy(config)

    # Set a single seed in the configuration
    config["seed"] = seed

    # Create directory for this seed
    path = os.path.join(base_path, f"seed_{seed}")
    assert not os.path.exists(path), f"found existing path '{path}', aborting setup"
    os.makedirs(path)

    # Save the configuration for this seed - will be loaded when experiment is run
    config_path = os.path.join(path, "config.yaml")
    with open(config_path, 'w') as config_file:
        yaml.dump({name: config}, config_file)

    return path


def setup_experiment(base_path, name, config):
    path = get_experiment_dir(base_path, name)

    # Save configuration - useful for offline hyperparameter tuning
    config_path = os.path.join(path, "config.yaml")
    with open(config_path, 'w') as config_file:
        yaml.dump({name: config}, config_file)
    
    # Get random seeds
    num_seeds = config.pop("num_seeds", 1)
    seeds = config.pop("seeds", list(range(num_seeds)))

    # Set up experiment directories for individual seeds
    return [setup_seed(path, name, config, seed) for seed in seeds]


def setup_experiments(configs,
                      output_path,
                      num_seeds=None, 
                      seeds=None, 
                      arguments=None):
    
    # Load config files (if necessary) and combine all configs into a single dictionary
    if isinstance(configs, str) or isinstance(configs, dict):
        configs = [configs]

    experiments = {}
    for config in configs:
        if isinstance(config, str):
            with open(config) as f:
                experiments.update(yaml.load(f, Loader=yaml.FullLoader))
        else:
            experiments.update(config)
    
    # Set up experiments
    paths = defaultdict(list)
    for name, config in experiments.items():

        # Override config if custom seeds are provided
        if num_seeds is not None:
            config["num_seeds"] = num_seeds
        
        if seeds is not None:
            config["seeds"] = seeds
        
        # Add custom command-line arguments to config if needed
        if arguments is not None:
            config["arguments"] = arguments

        # Get hyperparameter variations the config contains `grid_search` keys
        variations = grid_search(name, config)
    
        # Set up experiment directories
        if variations is None:
            paths[name] += setup_experiment(output_path, name, config)
        else:  # Hyperparameter sweep - setup subdirectories for each configuration
            base_path = get_experiment_dir(output_path, name)

            for var_name, var_config in variations.items():
                paths[name] += setup_experiment(base_path, var_name, var_config)

    return paths


def run_experiment(path,
                   device='cpu',
                   flush_secs=180,
                   csv=True,
                   verbose=True):

    # Check that the experiment directory exists
    assert os.path.isdir(path), f"No directory '{path}' exists"

    # Attempt to load the configuration file
    with open(os.path.join(path, "config.yaml")) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Get experiment name and config dictionary
    name, config = next(iter(config.items()))

    # Get the random seed for this trial
    assert "seed" in config, "No 'seed' field in config file"
    seed = config["seed"]

    # Get the maximum number of training iterations
    max_iterations = config.get("iterations", 100)

    # Print a message indicated we have started the experiment
    if verbose:
        print(f"Launching {name}, seed: {seed}, num iterations: {max_iterations}")

    # Build trainer - really this is an "experiment" rather than a "trainer"
    trainer_cls = get_experiment_class(config.get("trainer", "default"))
    trainer = trainer_cls(config.get("config", {}), seed=seed, device=device)

    # If requested, accumulate statistics and save to a CSV file
    stat_values = defaultdict(list)
    stat_indices = defaultdict(list)

    # Run trainer with TensorboardX logging
    with SummaryWriter(path, flush_secs=flush_secs) as writer:
        for iteration in range(max_iterations):
            stats = trainer.iterate()

            # Write statistics to tensorboard, and accumulate if requested
            for key, value in stats.items():
                writer.add_scalar(key, value, iteration)

                if csv:
                    stat_values[key].append(value)
                    stat_indices[key].append(iteration)

    # If requested, save results to a CSV file for more convenient analysis - do any of the analysis scripts use these?
    if csv:
        series = {}
        for key, values in stat_values.items():
            series[key] = pandas.Series(np.asarray(values), np.asarray(stat_indices[key]))

        dataframe = pandas.DataFrame(series)
        dataframe.to_csv(os.path.join(path, "results.csv"))

    # Allow the experiment to save any artifacts, such as strategies - drop this if we never use it
    artifact_path = os.path.join(path, "artifacts")
    trainer.save_artifacts(artifact_path)


def load_experiment(path):
    if not os.path.isdir(path):
        raise Exception(f"Experiment directory '{path}' does not exist")

    runs = []
    for obj in os.listdir(path):
        seed_path = os.path.join(path, obj)

        if os.path.isdir(seed_path):
            data = pandas.read_csv(os.path.join(seed_path, "results.csv"))

            # Filter out empy data series
            if data.shape[0] > 0:
                runs.append(data)

    return runs
