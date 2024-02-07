from abc import ABCMeta, abstractmethod
import numpy as np
import os.path
import torch

from hi_c.games import get_game_class
from hi_c.learners import get_learner_class
from hi_c.util import Stopwatch


class Experiment(metaclass=ABCMeta):
    """Abstract base class for iterative computational experiments."""

    @abstractmethod
    def iterate(self):
        raise NotImplementedError()

    @abstractmethod
    def save_artifacts(self, path):
        raise NotImplementedError()
    

class SimultaneousLearning(Experiment):
    """Multi-agent learning experiment in which players update their strategies
    simultaneously, given their partners' most recent strategies."""

    def __init__(self, config, seed, device):

        # Seed random number generators
        seq = np.random.SeedSequence(seed)
        rng = np.random.default_rng(seq)

        # Seed numpy and torch just in case the local numpy RNG is bypassed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Get the numer of gradient updates to perform per iteration
        self._steps_per_iter = config.get("steps_per_iter", 1)

        # Construct differentiable game
        assert "game" in config, "Must specify game through the 'game' field"
        game_cls = get_game_class(config["game"])
        game_config = config.get("game_config", {})
        self._game = game_cls(**game_config, device=device)

        # Construct learners
        assert "learners" in config, "Must provide learner configs via the 'learners' field"
        assert len(config["learners"]) >= len(self._game.strategy_spaces), "Not enough learners defined"

        self._learners = []
        self._strategies = []
        for pid, conf in enumerate(config["learners"]):
            cls = get_learner_class(conf.get("name", "naive"))  # Default to "naive" gradient ascent
            params = conf.get("params", {})
            learner = cls(self._game, pid, rng=rng, device=device, **params)
            self._learners.append(learner)
            self._strategies.append(learner.reset())

        # Initialize strategy history
        self._history = [[strategy.numpy(force=True)] for strategy in self._strategies]

        # Set up statistics
        self._timer = Stopwatch()
        self._total_steps = 0

    def iterate(self):
        self._timer.start()
        for _ in range(self._steps_per_iter):
            new_strategies = [learner.step(self._strategies) for learner in self._learners]
            self._strategies = new_strategies
        
        payoffs = self._game.payoffs(*self._strategies)
        self._timer.stop()

        # Save final joint strategy for each iteration
        for pid, strategy in enumerate(self._strategies):
            self._history[pid].append(strategy.numpy(force=True))

        # Return iteration statistics (the "global" key helps with tensorboard)
        self._total_steps += self._steps_per_iter
        stats = {
            "global/iteration_time": self._timer.latest,
            "global/total_time": self._timer.elapsed,
            "global/total_steps": self._total_steps
        }

        total_payoff = 0.
        for pid, payoff in enumerate(payoffs):
            payoff = payoff.item()  # Payoffs are scalar torch tensors
            total_payoff += payoff
            stats[f"global/payoff_{pid}"] = payoff
        
        stats["global/total_payoff"] = total_payoff
        return stats

    def save_artifacts(self, directory):
        os.makedirs(directory, exist_ok=True)
        for pid, strategies in enumerate(self._history):
            path = os.path.join(directory, f"strategies_{pid}")
            np.save(path, strategies, allow_pickle=False)


def get_experiment_class(name):
    if name != "default":
        raise ValueError(f"Experiment '{name}' is not defined")

    return SimultaneousLearning
