import torch

from hi_c.util import get_schedule
from hi_c.learners.gradient import GradientLearner

class LOLA(GradientLearner):
    """First-order LOLA (implementation for a single player)"""

    def __init__(self, 
                 game, 
                 id, 
                 other_lr=1, 
                 correction=True, 
                 **kwargs):
        super(LOLA, self).__init__(game, id, **kwargs)
        self._other_lr = get_schedule(other_lr)
        
        if correction:
            self._grad_fn = corrected_lola
        else:
            self._grad_fn = original_lola

    def gradient(self, strategies):
        payoffs = self.game.payoffs(*strategies)
        grads = torch.autograd.grad([payoffs[self.id]], strategies, create_graph=True)

        gradient = grads[self.id].detach()
        other_lr = self._other_lr.step()
        for id, grad in enumerate(grads):
            if id != self.id:
                term = self._grad_fn(self.strategy,strategies[id], payoffs[id], grad)
                gradient.add_(term, alpha=other_lr)
        
        return gradient


def original_lola(strategy,
                  other_strategy, 
                  other_payoff,
                  other_grad):
    grad, = torch.autograd.grad([other_payoff], [other_strategy], create_graph=True)
    grad, = torch.autograd.grad([grad], [strategy], grad_outputs=other_grad)
    return grad


def corrected_lola(strategy,
                   other_strategy, 
                   other_payoff,
                   other_grad):
        term, = torch.autograd.grad([other_payoff], [other_strategy], create_graph=True)
        term = torch.matmul(other_grad, term)

        grad, = torch.autograd.grad([term], [strategy])
        return grad
