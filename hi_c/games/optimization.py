import torch

from hi_c.util import Box

class Quadratic:

    def __init__(self,
                 dims=1,
                 players=1,
                 matrix=None,
                 device="cpu"):
        
        if matrix is not None:
            dims = matrix.shape[0]
            self._matrix = torch.tensor(matrix, dtype=torch.float, device=device)
        else:
            self._matrix = torch.eye(dims, dtype=torch.float, device=device)
        
        self.strategy_spaces = [Box((dims,)) for _ in range(players)]

    def payoffs(self, *params):
        payoffs = []
        for p in params:
            payoffs.append(-torch.matmul(p, torch.matmul(self._matrix, p)))
        
        return payoffs

    def payoffs(self, *params):
        payoffs = []
        for p in params:
            payoffs.append(-torch.matmul(p, torch.matmul(self._matrix, p)))
        
        return payoffs


class Gaussian:

    def __init__(self,
                 dims=1,
                 players=1,
                 scale=1.0,
                 matrix=None,
                 device="cpu"):
        self._scale = scale

        if matrix is not None:
            dims = matrix.shape[0]
            self._matrix = torch.tensor(matrix, dtype=torch.float, device=device)
        else:
            self._matrix = torch.eye(dims, dtype=torch.float, device=device)
        
        self.strategy_spaces = [Box((dims,)) for _ in range(players)]

    def payoffs(self, *params):
        payoffs = []
        for p in params:
            logits = torch.matmul(p, torch.matmul(self._matrix, p))
            payoffs.append(self._scale * torch.exp(-logits))
        
        return payoffs
