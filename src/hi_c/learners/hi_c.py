import torch

from hi_c.util import get_schedule

class HiC:
    """Uncoupled hierarchical gradients with simultaneous perturbations + commitments (ours)"""

    def __init__(self, game, config, rng, device):
        self._game = game
        self._rng = rng
        self._device = device

        self._lr = get_schedule(config.get("lr", 0.005))
        self._p = get_schedule(config.get("delta", 0.001))
        self._k = get_schedule(config.get("kappa", 10))
        self._initial = config.get("initial", None)

        self._strategy = None
        self._perturbation = None
        self._sampled_strategy = None

        self._counter = 0

    def _sample(self):
        perturbation = self._rng.integers(0, 1, size=self._game.strategy_spaces[0].shape)
        perturbation = 2 * perturbation - 1
        self._perturbation = torch.tensor(perturbation,
                                          dtype=torch.float32,
                                          device=self._device)

        self._sampled_strategy = self._strategy + self._p.step() * perturbation
        

    def reset(self):
        if self._initial is not None:
            self._strategy = self._initial
        else:
            self._strategy = self._game.strategy_spaces[0].sample(self._rng)
        
        self._strategy = torch.tensor(self._strategy,
                                      requires_grad=True, 
                                      dtype=torch.float,
                                      device=self._device)

        self._sample()
        self._counter = 0

        return self._sampled_strategy

    def step(self, other_strategy):
        self._counter += 1
        if self._counter >= self._k.step():
            self._counter = 0
            other_strategy = other_strategy.detach()
            
            payoff, _ = self._game.payoffs(self._strategy, other_strategy)
            grad = (payoff / self._p) / self._perturbation

            with torch.no_grad():
                self._strategy.add_(grad, alpha=self._lr)
                self._strategy.clamp_(self._game.strategy_spaces[0].min, 
                                    self._game.strategy_spaces[1].max)

            self._sample()

        return self._sampled_strategy
