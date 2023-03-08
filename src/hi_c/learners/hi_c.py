import numpy as np  # NOTE: We won't need this once the RNG is setup properly
import torch

class HiC:
    """Uncoupled hierarchical gradients with simultaneous perturbations + commitments (ours)"""

    def __init__(self, game, rng, lr=0.001, p=0.001, k=10, initial=None):  # NOTE: Need to implement learning-rate, commitment time schedules
        self._game = game
        self._rng = rng
        self._lr = lr
        self._p = p
        self._k = k
        self._initial = initial

        self._strategy = None
        self._perturbation = None
        self._sampled_strategy = None

        self._counter = 0

    def _sample(self):
        # NOTE: Fix random number generation
        perturbation = self._rng.integers(0, 1, size=self._game.strategy_spaces[0].shape)
        perturbation = 2 * perturbation - 1

        self._sampled_strategy = self._strategy.numpy(force=True)
        self._sampled_strategy += self._p * perturbation

        self._perturbation = torch.tensor(perturbation, dtype=torch.float32)

    def reset(self):
        if self._initial is not None:
            self._strategy = self._initial
        else:
            self._strategy = self._game.strategy_spaces[0].sample(self._rng)
        
        self._strategy = torch.tensor(self._strategy, requires_grad=True, dtype=torch.float)

        self._sample()
        self._counter = 0

        return self._sampled_strategy

    def step(self, other_strategy):
        self._counter += 1
        if self._counter >= self._k:
            self._counter = 0
            
            other_strategy = torch.tensor(other_strategy, requires_grad=True, dtype=torch.float32)
            payoff, _ = self._game.payoffs(self._strategy, other_strategy)
            grad = (payoff / self._p) / self._perturbation

            with torch.no_grad():
                self._strategy.add_(grad, alpha=self._lr)
                self._strategy.clamp_(self._game.strategy_spaces[0].min, 
                                    self._game.strategy_spaces[1].max)

            self._sample()

        return self._sampled_strategy, {}
