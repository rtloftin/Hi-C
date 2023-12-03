from hi_c.util import Box

class TandemGame:
    """The tandem bicycle game from the SOS paper (need to find the reference for this)"""

    def __init__(self, device="cpu"):
        self.strategy_spaces = [Box((1,)), Box((1,))]
    
    def payoffs(self, params_a, params_b):
        base = (params_a[0] + params_b[0]) ** 2
        return base - 2 * params_a[0], base - 2 * params_b[0]


class HamiltonianGame:
    """The Hamiltonian game (referenced in the SOS paper, but originally from a Balduzzi paper)"""

    def __init__(self, device="cpu"):
        self.strategy_spaces = [Box((1,)), Box((1,))]
    
    def payoffs(self, params_a, params_b):
        payoff = params_a[0] * params_b[0]
        return payoff, -payoff