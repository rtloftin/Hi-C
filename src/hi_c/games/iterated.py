import torch

from hi_c.util import Box

# Bi-matrix Games
PRISONERS_DILEMMA = [
    [
        [-1., -3.], 
        [ 0., -2.]
    ],
    [
        [-1.,  0.], 
        [-3., -2.]
    ]
]

MATCHING_PENNIES = [
    [
        [ 1., -1.], 
        [-1.,  1.]
    ],
    [
        [-1.,  1.], 
        [ 1., -1.]
    ]
]

POTENTIAL = [
    [
        [1., 2.], 
        [ 2., 4.]
    ],
    [
        [1.,  2.], 
        [2., 4.]
    ]
]

COORDINATION = [
    [
        [1., 0.], 
        [0., 1.]
    ],
    [
        [1., 0.], 
        [0., 1.]
    ]
]

ZERO = [
    [
        [0., 0.], 
        [0., 0.]
    ],
    [
        [0., 0.], 
        [0., 0.]
    ]
]

GAMES = {
    "prisoners_dilemma": PRISONERS_DILEMMA,
    "matching_pennies": MATCHING_PENNIES,
    "potential": POTENTIAL,
    "coordination": COORDINATION,
    "zero": ZERO
}

# TODO: Switch to sigmoid strategy spaces

# Pre-Defined Strategies
TIT_FOR_TAT = [
    [1., 1., 0., 1., 0.],
    [1., 1., 1., 0., 0.]
]

class IteratedGame:
    """Differentiable model of an infinitely repeated, discounted bimatrix game"""

    def __init__(self, name=None, payoffs=None, discount=0.9, l2=0.0, device="cpu"):
        self._discount = discount
        self._l2 = l2
        
        if name is not None:
            assert name in GAMES, f"Matrix game '{name}' is not defined"
            payoffs = GAMES[name]
        elif payoffs is None:
            raise ValueError("Must either specify a game by name, or provide payoff matrices")
        
        self._rewards = []
        for payoff in payoffs:
            r = [payoff[0][0], payoff[0][1], payoff[1][0], payoff[1][1]]
            self._rewards.append(torch.tensor(r, dtype=torch.float, device=device))

        self._eye = torch.eye(4, device=device)
        self.strategy_spaces = [Box((5,)), Box((5,))]

    def payoffs(self, strategy_a, strategy_b):

        # print(f"Strategy A: {strategy_a}, Strategy B: {strategy_b}")

        # Compute L2 penalty if needed
        if self._l2 > 0:
            penalty_a = self._l2 * torch.sum(strategy_a**2, dim=-1)
            penalty_b = self._l2 * torch.sum(strategy_b**2, dim=-1)

        # Compute probabilities from logits
        strategy_a = torch.sigmoid(strategy_a)
        strategy_b = torch.sigmoid(strategy_b)

        # Split into initial distribution and transition probabilities
        a_0 = strategy_a[0]
        b_0 = strategy_b[0]
        a = strategy_a[1:]
        b = strategy_b[1:]

        # Initial state distribution
        p_0 = [a_0 * b_0, a_0 * (1 - b_0), (1 - a_0) * b_0, (1 - a_0) * (1 - b_0)]
        p_0 = torch.stack(p_0, dim=-1)
        
        # Transition matrix
        P = [a * b, a * (1 - b), (1 - a) * b, (1 - a) * (1 - b)]
        P = torch.stack(P, dim=-1)

        # Compute state distribution
        inverse = torch.linalg.inv(self._eye - (self._discount * P))
        D = torch.matmul(p_0, inverse)

        # Compute payoffs
        return_a = (1-self._discount) * torch.matmul(D, self._rewards[0])
        return_b = (1-self._discount) * torch.matmul(D, self._rewards[1])

        # Add penalties if needed
        if self._l2 > 0:
            return_a = return_a - penalty_a
            return_b = return_b - penalty_b

        return return_a, return_b
