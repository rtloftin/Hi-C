from abc import ABCMeta, abstractmethod
import numpy as np
import torch

from hi_c.util import get_schedule

class GradientLearner(metaclass=ABCMeta):
    
    def __init__(self, 
                 game, 
                 id, 
                 lr=0.005, 
                 std=0.5, 
                 rng=None, 
                 device="cpu"):
        self.id = id
        self.game = game
        self.std = std
        self.device = device
        self.strategy = None
 
        # Get parameter space
        self.space = game.strategy_spaces[id]

        # Construct random number generator if none provided
        self.rng = rng if rng is not None else np.random.default_rng()

        # Configure learning-rate schedule
        self.lr = get_schedule(lr)
    
    @abstractmethod
    def gradient(self, strategies):
        raise NotImplementedError()

    def reset(self):
        if self.std > 0:
            initial = self.rng.normal(scale=self.std, size=self.space.shape)
        else:
            initial = np.zeros(self.space.shape)
        
        initial = initial.clip(self.space.min, self.space.max)
        self.strategy = torch.tensor(initial, 
                                     requires_grad=True, 
                                     dtype=torch.float,
                                     device=self.device)
        return self.strategy

    def step(self, strategies):
        detached = []
        for id, strategy in enumerate(strategies):
            if id != self.id:
                detached.append(strategy.detach().requires_grad_(True))
            else:
                detached.append(self.strategy)

        gradient = self.gradient(detached)

        with torch.no_grad():
            self.strategy.add_(gradient, alpha=self.lr.step())
            self.strategy.clamp_(self.space.min, self.space.max)

        return self.strategy


class NaiveLearner(GradientLearner):
    """Gradient ascent learner"""

    def __init__(self, game, id, **kwargs):
        super(NaiveLearner, self).__init__(game, id, **kwargs)

    def gradient(self, strategies):
        payoffs = self.game.payoffs(*strategies)
        gradient, = torch.autograd.grad([payoffs[self.id]], [self.strategy])

        return gradient
