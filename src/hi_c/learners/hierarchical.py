import torch

from hi_c.util import get_schedule
from hi_c.learners.gradient import GradientLearner

class HierarchicalGradient:
    """Hierarchical gradient ascent (Fiez et al. 2019)"""

    def __init__(self, game, **kwargs):
        super(HierarchicalGradient, self).__init__(game, **kwargs)
    
    def _gradient(self, other_strategy):
        payoff_a, payoff_b = self.game.payoffs(self.strategy, other_strategy)
        grad_a, grad_b = torch.autograd.grad([payoff_a], [self.strategy, other_strategy], retain_graph=True)

        grad, = torch.autograd.grad([payoff_b], [other_strategy], create_graph=True)
        
        hessian = []
        for x in grad:
            hessian.append(torch.autograd.grad([x], [other_strategy], retain_graph=True)[0])
        
        hessian = torch.stack(hessian, dim=-2)
        hessian = hessian.detach()
        vec = torch.matmul(grad_b, torch.linalg.inv(hessian))

        grad_r, = torch.autograd.grad([grad], [self.strategy], grad_outputs=vec)

        return grad_a - grad_r
