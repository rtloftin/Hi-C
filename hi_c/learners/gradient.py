from abc import ABCMeta, abstractmethod
import numpy as np
import torch

from hi_c.learners.schedule import get_schedule


class GradientLearner(metaclass=ABCMeta):  # Why is this an abstract class?  Apparently LOLA also implements this
    """
    Abstract base class for all gradient-based learners.  Different learners such as LOLA and Hierarchical learner
    implement different versions of the `gradient()` function.
    """

    def __init__(self, 
                 game,
                 player_id,
                 lr=0.005, 
                 initialization_std=0.5,
                 rng=None,  # NOTE: This is how we do random seeding now
                 device="cpu"):
        self.game = game
        self.player_id = player_id
        self.initialization_std = initialization_std
        self.device = device
        self.strategy = None
 
        # Get parameter space
        self.space = game.strategy_spaces[player_id]

        # Construct random number generator if none provided
        self.rng = rng if rng is not None else np.random.default_rng()

        # Configure learning-rate schedule
        self.lr = get_schedule(lr)
    
    @abstractmethod
    def gradient(self, strategies):
        raise NotImplementedError()

    def reset(self):
        if self.initialization_std > 0.:  # NOTE: Initialization handled by numpy, not torch
            # Need a different initializer for the cournot game, clipping to zero - how did we fix this? - seems we didn't, starting at zero does work
            # May want to implement sampling from the strategy space, something Gym already does
            initial = self.rng.normal(scale=self.initialization_std, size=self.space.shape)
        else:
            initial = np.zeros(self.space.shape)
        
        # initial = initial.clip(self.space.min, self.space.max)  # NOTE: We turned this off, wouldn't that create issues for the Cournot game?
        self.strategy = torch.tensor(initial, 
                                     requires_grad=True, 
                                     dtype=torch.float,
                                     device=self.device)
        return self.strategy

    def step(self, strategies):
        detached = []
        for player_id, strategy in enumerate(strategies):
            if player_id != self.player_id:
                detached.append(strategy.detach().requires_grad_(True))
            else:
                detached.append(self.strategy)  # Our strategy is not detached from the graph

        gradient = self.gradient(detached)

        with torch.no_grad():
            self.strategy.add_(gradient, alpha=self.lr.step())
            # self.strategy.clamp_(self.space.min, self.space.max)

        return self.strategy


class NaiveLearner(GradientLearner):
    """
    Simple gradient ascent learner.  Performs gradient ascent on the player's individual payoff function, treating
    other agents' actions as constants.
    """

    def __init__(self, game, player_id, **kwargs):
        super(NaiveLearner, self).__init__(game, player_id, **kwargs)

    def gradient(self, strategies):
        payoffs = self.game.payoffs(*strategies)
        gradient, = torch.autograd.grad([payoffs[self.player_id]], [self.strategy])

        return gradient
