from abc import ABCMeta, abstractmethod
import numpy as np
import torch

from hi_c.util import get_schedule

class GradientLearner(metaclass=ABCMeta):
    
    def __init__(self, game, lr=0.005, std=0.5, rng=None, device="cpu"):
        self.game = game
        self.std = std
        self.device = device
        self.strategy = None

        # Construct random number generator if none provided
        self.rng = rng if rng is not None else np.random.default_rng()

        # Configure learning-rate schedule
        self.lr = get_schedule(lr)
    
    @abstractmethod
    def _gradient(self, other_strategy):
        raise NotImplementedError()

    def reset(self):
        shape = self.game.strategy_spaces[0].shape
        if self.std > 0:
            initial = self.rng.normal(scale=self.std, size=shape)
        else:
            initial = np.zeros(shape)
        
        self.strategy = torch.tensor(initial, 
                                     requires_grad=True, 
                                     dtype=torch.float,
                                     device=self.device)
        return self.strategy

    def step(self, other_strategy):
        gradient = self._gradient(other_strategy.detach().requires_grad_(True))

        with torch.no_grad():
            self.strategy.add_(gradient, alpha=self.lr.step())
            self.strategy.clamp_(self.game.strategy_spaces[0].min, 
                                 self.game.strategy_spaces[1].max)

        return self.strategy


class NaiveLearner(GradientLearner):
    """Gradient ascent learner"""

    def __init__(self, game, **kwargs):
        super(NaiveLearner, self).__init__(game, **kwargs)

    def _gradient(self, other_strategy):
        payoff, _ = self.game.payoffs(self.strategy, other_strategy)
        gradient, = torch.autograd.grad([payoff], [self.strategy])

        return gradient
