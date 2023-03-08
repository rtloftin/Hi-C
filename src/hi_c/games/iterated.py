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

GAMES = {
    "prisoners_dilemma": PRISONERS_DILEMMA,
    "matching_pennies": MATCHING_PENNIES,
    "potential": POTENTIAL,
    "coordination": COORDINATION,
}

# Pre-Defined Strategies
TIT_FOR_TAT = [
    [1., 1., 0., 1., 0.],
    [1., 1., 1., 0., 0.]
]

class IteratedGame:
    """Differentiable model of an infinitely repeated, discounted bimatrix game"""

    def __init__(self, config, device):
        self._discount = config.get("discount", 0.9)
        
        if "name" in config:
            assert config["name"] in GAMES, f"Matrix game '{config['name']}' is not defined"
            payoffs = GAMES[config["name"]]
        elif "payoffs" in config:
            payoffs = config["payoffs"]
        else:
            raise ValueError("Must either specify a game by name, or provide payoff matrices")
        
        self._rewards = []
        for payoff in payoffs:
            r = [payoff[0][0], payoff[0][1], payoff[1][0], payoff[1][1]]
            self._rewards.append(torch.tensor(r, dtype=torch.float, device=device))

        self._eye = torch.eye(4, device=device)

        epsilon = config.get("epsilon", 0.0001)
        self.strategy_spaces = [
            Box(epsilon, 1. - epsilon, (5,)),
            Box(epsilon, 1. - epsilon, (5,))
        ]

    def payoffs(self, strategy_a, strategy_b):
        
        # Split parameter vectors
        a_0 = strategy_a[0]
        b_0 = strategy_b[0]
        a = strategy_a[1:]
        b = strategy_b[1:]

        # Initial state distribution
        p_0 = [None] * 4
        p_0[0] = a_0 * b_0
        p_0[1] = a_0 * (1 - b_0)
        p_0[2] = (1 - a_0) * b_0
        p_0[3] = (1 - a_0) * (1 - b_0)
        p_0 = torch.stack(p_0)
        
        # Transition matrix
        P = [a * b, a * (1 - b), (1 - a) * b, (1 - a) * (1 - b)]
        P = torch.stack(P, dim=1)

        # Compute state distribution
        inverse = torch.linalg.inv(self._eye - (self._discount * P))
        D = torch.matmul(p_0, inverse)

        # Compute payoffs
        returns = []
        for reward in self._rewards:
            returns.append(torch.matmul(D, reward))

        return returns
