from abc import ABCMeta, abstractmethod
import numpy as np
import torch

from hi_c.learners.schedule import get_schedule


class GradientLearner(metaclass=ABCMeta):
    """Abstract base class for all gradient-based learners.  Different learners such as LOLA and the Hierarchical
    learner implement different versions of the `gradient()` function."""

    def __init__(self, 
                 game,
                 player_id,
                 lr=0.005,
                 rng=None,
                 device="cpu"):
        self.game = game
        self.player_id = player_id
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
        initial = self.space.sample(self.rng)
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
