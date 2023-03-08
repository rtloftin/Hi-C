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
    

class PairedTrainer(Trainable):

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
        self._game = game_cls(config.get("game_config", {}), device)

        # Construct learner A
        assert "learner_a" in config, "Must specify class for learner A through 'learner_a' field"
        learner_a_cls = get_learner_class(config["learner_a"])
        self._learner_a = learner_a_cls(self._game,
                                        config.get("learner_a_config", {}), 
                                        rng=rng, 
                                        device=device)

        # Construct learner B
        assert "learner_b" in config, "Must specify class for learner B through 'learner_b' field"
        learner_b_cls = get_learner_class(config["learner_b"])
        self._learner_b = learner_b_cls(ReversedGame(self._game),
                                        config.get("learner_b_config", {}), 
                                        rng=rng, 
                                        device=device)

        # Initialize learners
        self._strategy_a = self._learner_a.reset()
        self._strategy_b = self._learner_b.reset()

        # Set up statistics
        self._timer = Stopwatch()
        self._strategies_a = []
        self._strategies_b = []
        self._total_steps = 0

    def train(self):
        self._timer.start()
        for _ in range(self._iteration_updates):
            new_strategy_a = self._learner_a.step(self._strategy_b)
            new_strategy_b = self._learner_b.step(self._strategy_a)
            self._strategy_a = new_strategy_b
            self._strategy_b = new_strategy_a
        
        payoff_a, payoff_b = self._game.payoffs(self._strategy_a, self._strategy_b)
        self._timer.stop()

        # Save final joint strategy for each iteration
        self._strategies_a.append(self._strategy_a.numpy(force=True))
        self._strategies_b.append(self._strategy_b.numpy(force=True))

        # Return iteration statistics
        self._total_steps += self._iteration_updates
        payoff_a = payoff_a.item()
        payoff_b = payoff_b.item()

        return {
            "payoff_a": payoff_a,
            "payoff_b": payoff_b,
            "total_payoff": payoff_a + payoff_b,
            "iteration_time": self._timer.latest,
            "total_time": self._timer.elapsed,
            "total_steps": self._total_steps
        }

    def save_checkpoint(self, dir):
        raise NotImplementedError()

    def save_artifacts(self, dir):
        strategies_a = np.stack(self._strategies_a)
        strategies_b = np.stack(self._strategies_b)

        np.save(os.path.join(dir, "strategies_a"), strategies_a, allow_pickle=False)
        np.save(os.path.join(dir, "strategies_b"), strategies_b, allow_pickle=False)


def get_trainer_class(name):
    if name != "default":
        raise ValueError(f"Trainer {name} is not defined")

    return PairedTrainer
