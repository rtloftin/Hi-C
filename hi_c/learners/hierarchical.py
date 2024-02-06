import torch

from hi_c.learners.gradient import GradientLearner


class HierarchicalGradient(GradientLearner):
    """Hierarchical gradient ascent (Fiez et al., 2019)"""

    def __init__(self, game, player_id, **kwargs):
        super(HierarchicalGradient, self).__init__(game, player_id, **kwargs)
    
    def gradient(self, strategies):
        assert len(strategies) == 2, "Hierarchical gradient learner only supports two-player games"
        other_id = (self.player_id + 1) % 2
        other_strategy = strategies[other_id]

        payoffs = self.game.payoffs(*strategies)
        grad_a, grad_b = torch.autograd.grad([payoffs[self.player_id]],
                                             [self.strategy, other_strategy],
                                             retain_graph=True)

        grad, = torch.autograd.grad([payoffs[other_id]], [other_strategy], create_graph=True)
        
        hessian = []
        for x in grad:
            hessian.append(torch.autograd.grad([x], [other_strategy], retain_graph=True)[0])
        
        hessian = torch.stack(hessian, dim=-2)
        hessian = hessian.detach()
        vec = torch.matmul(grad_b, torch.linalg.inv(hessian))

        grad_r, = torch.autograd.grad([grad], [self.strategy], grad_outputs=vec)

        return grad_a - grad_r
