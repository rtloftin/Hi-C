import torch

from hi_c.util import get_schedule

class LOLA:
    """First-order LOLA (implementation for a single player)"""

    def __init__(self, game, config, rng, device):
        self._game = game
        self._rng = rng
        self._device = device

        self._lr = get_schedule(config.get("lr", 0.005))
        self._other_lr = get_schedule(config.get("other_lr", 1))
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
        other_strategy = torch.tensor(other_strategy,
                                      requires_grad=True, 
                                      dtype=torch.float,
                                      device=self._device)

        payoff_a, payoff_b = self._game.payoffs(self._strategy, other_strategy)
        grad_a, grad_b = torch.autograd.grad([payoff_a], [self._strategy, other_strategy], retain_graph=True)

        grad, = torch.autograd.grad([payoff_b], [other_strategy], create_graph=True)
        grad, = torch.autograd.grad([grad], [self._strategy], grad_outputs=grad_b)

        with torch.no_grad():
            self._strategy.add_(grad_a + self._other_lr * grad, alpha=self._lr)
            self._strategy.clamp_(self._game.strategy_spaces[0].min, 
                                  self._game.strategy_spaces[1].max)

        return self._strategy
    