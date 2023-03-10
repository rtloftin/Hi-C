import torch

from hi_c.util import get_schedule
from hi_c.learners.gradient import GradientLearner

class LOLA(GradientLearner):
    """First-order LOLA (implementation for a single player)"""

    def __init__(self, game, other_lr=1, correction=True, **kwargs):
        super(LOLA, self).__init__(game, **kwargs)
        self._other_lr = get_schedule(other_lr)
        self._correction = correction
    
    def _original_gradient(self, other_strategy):
        payoff_a, payoff_b = self.game.payoffs(self.strategy, other_strategy)
        grad_a, grad_b = torch.autograd.grad([payoff_a], [self.strategy, other_strategy], retain_graph=True)

        grad, = torch.autograd.grad([payoff_b], [other_strategy], create_graph=True)
        grad, = torch.autograd.grad([grad], [self.strategy], grad_outputs=grad_b)
        
        return grad_a + self._other_lr.step() * grad
    
    def _corrected_gradient(self, other_strategy):
        payoff_a, payoff_b = self.game.payoffs(self.strategy, other_strategy)
        grad_a, grad_b = torch.autograd.grad([payoff_a], [self.strategy, other_strategy], create_graph=True)

        term, = torch.autograd.grad([payoff_b], [other_strategy], create_graph=True)
        term = torch.matmul(grad_b, term)

        grad, = torch.autograd.grad([term], [self.strategy])
        
        return grad_a + self._other_lr.step() * grad

    def _gradient(self, other_strategy):
        if self._correction:
            return self._corrected_gradient(other_strategy)
        else:
            return self._original_gradient(other_strategy)
