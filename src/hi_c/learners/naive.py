import torch

class NaiveLearner:
    """Gradient ascent learner"""

    def __init__(self, game, rng, lr=0.001, initial=None):
        self._game = game
        self._rng = rng
        self._lr = lr
        self._initial = initial
        self._strategy = None

    def reset(self):
        if self._initial is not None:
            self._strategy = self._initial
        else:
            self._strategy = self._game.strategy_spaces[0].sample(self._rng)
        
        self._strategy = torch.tensor(self._strategy, requires_grad=True, dtype=torch.float)
        return self._strategy.numpy(force=True)

    def step(self, other_strategy):
        other_strategy = torch.tensor(other_strategy, requires_grad=True, dtype=torch.float32)

        payoff, _ = self._game.payoffs(self._strategy, other_strategy)
        gradient, = torch.autograd.grad([payoff], [self._strategy])

        with torch.no_grad():
            self._strategy.add_(gradient, alpha=self._lr)
            self._strategy.clamp_(self._game.strategy_spaces[0].min, 
                                  self._game.strategy_spaces[1].max)

        return self._strategy.numpy(force=True), {}
