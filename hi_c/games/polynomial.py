from hi_c.util import Box


class TandemGame:
    """
    The tandem bicycle game (Letcher et al., ICLR 2019)

    Payoff functions:
        f_1(x, y) = (x + y)^2 + 2x
        f_2(x, y) = (x + y)^2 + 2y
    """

    def __init__(self, device="cpu"):
        self.strategy_spaces = [
            Box(-4., 4., (1,)),
            Box(-4., 4., (1,))
        ]
    
    def payoffs(self, params_a, params_b):
        base = (params_a[0] + params_b[0]) ** 2
        return 2 * params_a[0] - base, 2 * params_b[0] - base


class HamiltonianGame:
    """
    The Hamiltonian game (Letcher et al., ICLR 2019; Fiez et al., ICML 2020)

    Payoff functions:
        f_1(x, y) = xy
        f_2(x, y) = -xy
    """

    def __init__(self, device="cpu"):
        self.strategy_spaces = [
            Box(-4., 4., (1,)),
            Box(-4., 4., (1,))
        ]
    
    def payoffs(self, params_a, params_b):
        payoff = params_a[0] * params_b[0]
        return payoff, -payoff
