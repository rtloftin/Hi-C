import torch

from hi_c.util import get_schedule

class NaiveLearner:
    """Gradient ascent learner"""

    def __init__(self, game, config, rng, device):
        self._game = game
        self._rng = rng
        self._device = device

        self._lr = get_schedule(config.get("lr", 0.005))
        self._initial = config.get("initial", None)
        self._strategy = None

    def reset(self):
        if self._initial is not None:
            self._strategy = self._initial
        else:
            self._strategy = self._game.strategy_spaces[0].sample(self._rng)
        
        self._strategy = torch.tensor(self._strategy, 
                                      requires_grad=True, 
                                      dtype=torch.float,
                                      device=self._device)
        return self._strategy

    def step(self, other_strategy):
        other_strategy = other_strategy.detach()

        payoff, _ = self._game.payoffs(self._strategy, other_strategy)
        gradient, = torch.autograd.grad([payoff], [self._strategy])

        with torch.no_grad():
            self._strategy.add_(gradient, alpha=self._lr.step())
            self._strategy.clamp_(self._game.strategy_spaces[0].min, 
                                  self._game.strategy_spaces[1].max)

        return self._strategy
