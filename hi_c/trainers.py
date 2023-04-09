from abc import ABCMeta, abstractmethod
import numpy as np
import os.path
import torch

from hi_c.games import get_game_class
from hi_c.learners import get_learner_class
from hi_c.util import ReversedGame, Stopwatch

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
    

class SimultaneousTrainer(Trainable):

    def __init__(self, config, seed, device):

        # Seed random number generators
        np.random.seed(seed)  # NOTE: Cannot be sure all environments use the seed we give them
        torch.manual_seed(seed)

        seq = np.random.SeedSequence(seed)
        rng = np.random.default_rng(seq)

        # Get the numer of gradient updates to perform per iteration
        self._iteration_updates = config.get("iteration_updates", 10)

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
        for id, conf in enumerate(config["learners"]):
            cls = get_learner_class(conf.get("name", "naive"))
            params = conf.get("params", {})
            learner = cls(self._game, id, rng=rng, device=device, **params)
            self._learners.append(learner)
            self._strategies.append(learner.reset())

        # Initialize history
        self._history = [[strategy.numpy(force=True)] for strategy in self._strategies]

        # Set up statistics
        self._timer = Stopwatch()
        self._total_steps = 0

    def train(self):
        self._timer.start()
        for _ in range(self._iteration_updates):
            new_strategies = [learner.step(self._strategies) for learner in self._learners]
            self._strategies = new_strategies
        
        payoffs = self._game.payoffs(*self._strategies)
        self._timer.stop()

        # Save final joint strategy for each iteration
        for id, strategy in enumerate(self._strategies):
            self._history[id].append(strategy.numpy(force=True))

        # Return iteration statistics
        self._total_steps += self._iteration_updates
        stats = {
            "global/iteration_time": self._timer.latest,
            "global/total_time": self._timer.elapsed,
            "global/total_steps": self._total_steps
        }

        total_payoff = 0
        for id, payoff in enumerate(payoffs):
            payoff = payoff.item()
            total_payoff += payoff
            stats[f"global/payoff_{id}"] = payoff
        
        stats["global/total_payoff"] = total_payoff
        return stats

    def save_checkpoint(self, dir):
        raise NotImplementedError()

    def save_artifacts(self, dir):
        for id, strategies in enumerate(self._history):
            path = os.path.join(dir, f"strategies_{id}")
            np.save(path, strategies, allow_pickle=False)


def get_trainer_class(name):
    if name != "default":
        raise ValueError(f"Trainer {name} is not defined")

    return SimultaneousTrainer
