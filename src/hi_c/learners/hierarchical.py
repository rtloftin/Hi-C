import torch

class HierarchicalGradient:
    """Hierarchical gradient ascent (Fiez et al. 2019)"""

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

        payoff_a, payoff_b = self._game.payoffs(self._strategy, other_strategy)
        grad_a, grad_b = torch.autograd.grad([payoff_a], [self._strategy, other_strategy], retain_graph=True)

        grad, = torch.autograd.grad([payoff_b], [other_strategy], create_graph=True)
        hessian = []

        for x in grad:
            hessian.append(torch.autograd.grad([x], [other_strategy], retain_graph=True)[0])
        
        hessian = torch.stack(hessian, dim=-2)
        hessian = hessian.detach()
        vec = torch.matmul(grad_b, torch.linalg.inv(hessian))

        grad_r, = torch.autograd.grad([grad], [self._strategy], grad_outputs=vec)

        with torch.no_grad():
            self._strategy.add_(grad_a - grad_r, alpha=self._lr)
            self._strategy.clamp_(self._game.strategy_spaces[0].min, 
                                  self._game.strategy_spaces[1].max)

        return self._strategy.numpy(force=True), {}
