import numpy as np
import torch

from hi_c.util import get_schedule

class HiC:
    """Uncoupled hierarchical gradients with simultaneous perturbations + commitments (ours)"""

    def __init__(self, 
                 game,
                 id,
                 lr=0.005, 
                 p=0.001, 
                 k=10, 
                 std=0.5, 
                 rng=None, 
                 device="cpu"):
        self._game = game
        self._id = id
        self._std = std
        self._device = device

        # Get parameter space
        self._space = game.strategy_spaces[id]

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
        perturbation = self._rng.integers(0, 2, size=self._space.shape)
        perturbation = 2 * perturbation - 1
        self._perturbation = torch.tensor(perturbation,
                                          requires_grad=False,
                                          dtype=torch.float32,
                                          device=self._device)

        self._last_p = self._p.step()
        self._sampled_strategy = self._strategy + self._last_p * self._perturbation
        self._sampled_strategy.clamp_(self._space.min, self._space.max)
    
    def reset(self):
        if self._std > 0:
            initial = self._rng.normal(scale=self._std, size=self._space.shape)
        else:
            initial = np.zeros(self._space.shape)
        
        initial = initial.clip(self._space.min, self._space.max)
        self._strategy = torch.tensor(initial, 
                                      requires_grad=False, 
                                      dtype=torch.float,
                                      device=self._device)

        self._sample()
        self._counter = 0

        return self._sampled_strategy

    def step(self, strategies):
        self._counter += 1
        if self._counter >= self._k.step():
            detached = []
            for id, strategy in enumerate(strategies):
                if id != self._id:
                    detached.append(strategy.detach())
                else:
                    detached.append(self._strategy)
            
            payoffs = self._game.payoffs(*detached)
            grad = (payoffs[self._id] / self._last_p) * self._perturbation

            self._strategy.add_(grad, alpha=self._lr.step())
            self._strategy.clamp_(self._space.min, self._space.max)

            self._sample()
            self._counter = 0

        return self._sampled_strategy
