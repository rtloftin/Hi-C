from abc import ABCMeta, abstractmethod
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from git import Repo
import numpy as np
import os
import os.path
import pandas
from tensorboardX import SummaryWriter
import yaml

from hi_c.grid_search import grid_search
from hi_c.trainers import get_trainer_class


def timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="minutes")


def git_commit():
    try:
        repo = Repo(search_parent_directories=True)
        return str(repo.active_branch.commit)
    except:
        return "unknown"


class Trainable(metaclass=ABCMeta):
    """Abstract base class for iterative computational experiments."""

    @abstractmethod
    def train(self):
        raise NotImplementedError()
    
    @abstractmethod
    def save_checkpoint(self, dir):
        raise NotImplementedError()

    @abstractmethod
    def save_artifacts(self, dir):
        raise NotImplementedError()


# TODO: Add support for loading from checkpoints
def run_experiment(path,
                   device='cpu',
                   verbose=False, 
                   flush_secs=200):

    # Check that the directory exists
    assert os.path.isdir(path), f"No directory '{path}' exists"

    # Attempt to load the configuration file
    with open(os.path.join(path, "config.yaml")) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Get experiment name and config dictionary
    name, config = next(iter(config.items()))

    # Get the random seed for this trial
    assert "seed" in config, "No 'seed' field in config file"
    seed = config["seed"]

    # Print a message indicated we have started the experiment
    print(f"Launching {name}, seed {seed}, from path: {path}")

    # Update run log for experiment
    with open(os.path.join(path, "run_log"), "a") as log_file:
        log_file.write(f"Experiment started at {timestamp()}, git commit: {git_commit()}")

    # TODO: Support training dependent termination criteria
    # Get the maximum number of training iterations
    max_iterations = config.get("iterations", 100)

    # Build trainer
    trainer_cls = get_trainer_class(config.get("trainer", "default"))
    trainer = trainer_cls(config.get("config", {}), seed=seed, device=device)

    # Run trainer with TensorboardX logging
    stat_values = defaultdict(list)
    stat_indices = defaultdict(list)
    iteration = 0
    complete = False
    
    with SummaryWriter(path, flush_secs=flush_secs) as writer:
        while not complete:
            stats = trainer.train()

            # TODO: Flatten stats dicts automatically
            # Write statistics to tensorboard and append to data series
            for key, value in stats.items():
                writer.add_scalar(key, value, iteration)
                stat_values[key].append(value)
                stat_indices[key].append(iteration)

            # TODO: Support for checkpointing

            # Check termination conditions
            iteration += 1
            if iteration >= max_iterations:
                complete = True

    # Build and save data frame for easier analysis
    series = {}
    for key, values in stat_values.items():
        series[key] = pandas.Series(np.asarray(values), np.asarray(stat_indices[key]))

    dataframe = pandas.DataFrame(series)
    dataframe.to_csv(os.path.join(path, "results.csv"))

    # TODO: Allow intermediate artifacts to be saved
    # Save any artifacts
    artifact_path = os.path.join(path, "artifacts")
    os.makedirs(artifact_path, exist_ok=True)
    trainer.save_artifacts(artifact_path)


def get_experiment_dir(base_path, name, index_digits=3):
    path = os.path.join(base_path, name + "_" + timestamp())

    idx = 0
    while os.path.exists(path):
        idx += 1
        path = os.path.join(base_path, name + "_" + str(idx).zfill(index_digits))
    
    os.makedirs(path)
    return path


def setup_seed(base_path, name, config, seed):
    config = deepcopy(config)

    # Set a single seed in the configuration
    del config["num_seeds"]
    del config["seeds"]
    config["seed"] = seed

    # Create directory for this seed
    path = os.path.join(base_path, f"seed_{seed}")
    assert not os.path.exists(path), f"found existing path '{path}', aborting setup"
    os.makedirs(path)

    # Save the configuration for this seed
    config_path = os.path.join(path, "config.yaml")
    with open(config_path, 'w') as config_file:
        yaml.dump({name: config}, config_file)

    return path


def setup_experiment(base_path, name, config):
    path = get_experiment_dir(base_path, name)

    # Save configuration - used for hyperparameter tuning
    config_path = os.path.join(path, "config.yaml")
    with open(config_path, 'w') as config_file:
        yaml.dump({name: config}, config_file)
    
    # Get random seeds
    num_seeds = config.get("num_seeds", 1)
    seeds = config.get("seeds", list(range(num_seeds)))

    # Set up individual seeds
    return [setup_seed(path, name, config, seed) for seed in seeds]


def setup_experiments(config_files,
                      output_path,
                      num_seeds=None, 
                      seeds=None, 
                      arguments=None):
    
    # Load config files and combine them into a single dictionary
    if isinstance(config_files, str):
        config_files = [config_files]

    experiments = {}
    for path in config_files:
        with open(path) as f:
            experiments.update(yaml.load(f, Loader=yaml.FullLoader))
    
    # Set up experiments
    paths = []
    for name, config in experiments.items():

        # Make experiment directory

        # Override config if custom seeds are provided
        if num_seeds is not None:
            config["num_seeds"] = num_seeds
        
        if seeds is not None:
            config["seeds"] = seeds
        
        # Add custom arguments to config if needed
        if arguments is not None:
            config["arguments"] = arguments

        # Get hyperparameter variations if specified
        variations = grid_search(name, config)
    
        # Set up experiment directories
        if variations is None:
            paths += setup_experiment(output_path, name, config)
        else:  # NOTE: Hyperparameter sweep
            base_path = get_experiment_dir(output_path, name)

            for var_name, var_config in variations.items():
                paths += setup_experiment(base_path, var_name, var_config)

    return paths
