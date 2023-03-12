import numpy as np
import torch

from hi_c.util import get_schedule

class HiC:
    """Uncoupled hierarchical gradients with simultaneous perturbations + commitments (ours)"""

    def __init__(self, 
                 game, 
                 lr=0.005, 
                 p=0.001, 
                 k=10, 
                 std=0.5, 
                 rng=None, 
                 device="cpu"):
        self._game = game
        self._std = std
        self._device = device

        # Construct random number generator if none provided
        self._rng = rng if rng is not None else np.random.default_rng()

        # Configure learning-rate, perturbation, and commitment schedules
        self._lr = get_schedule(lr)
        self._p = get_schedule(p)
        self._k = get_schedule(k)

        self._strategy = None
        self._sampled_strategy = None
        self._perturbation = None
        self._last_p = None

        self._counter = 0

    def _sample(self):
        perturbation = self._rng.integers(0, 2, size=self._game.strategy_spaces[0].shape)
        perturbation = 2 * perturbation - 1
        self._perturbation = torch.tensor(perturbation,
                                          dtype=torch.float32,
                                          device=self._device)

        self._last_p = self._p.step()
        self._sampled_strategy = self._strategy + self._last_p * self._perturbation
    
    def reset(self):
        shape = self._game.strategy_spaces[0].shape
        if self._std > 0:
            initial = self._rng.normal(scale=self._std, size=shape)
        else:
            initial = np.zeros(shape)
        
        self._strategy = torch.tensor(initial, 
                                      requires_grad=True, 
                                      dtype=torch.float,
                                      device=self._device)

        self._sample()
        self._counter = 0

        return self._sampled_strategy

    def step(self, other_strategy):
        self._counter += 1
        if self._counter >= self._k.step():
            other_strategy = other_strategy.detach()
            self._counter = 0
            
            payoff, _ = self._game.payoffs(self._strategy, other_strategy)
            grad = (payoff / self._last_p) * self._perturbation

            with torch.no_grad():
                self._strategy.add_(grad, alpha=self._lr.step())
                self._strategy.clamp_(self._game.strategy_spaces[0].min, 
                                      self._game.strategy_spaces[1].max)

            self._sample()

        return self._sampled_strategy
