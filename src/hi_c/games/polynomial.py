from hi_c.util import Box

class TandemGame:
    """The tandem bicycle game from the SOS paper (need to find the reference for this)"""

    def __init__(self, device="cpu"):
        self.strategy_spaces = [Box((1,)), Box((1,))]
    
    def payoffs(self, strategy_a, strategy_b):
        base = (strategy_a[0] + strategy_b[0]) ** 2
        return base - 2 * strategy_a[0], base - 2 * strategy_b[0]


class HamiltonianGame:
    """The Hamiltonian game (referenced in the SOS paper, but orgininally from a Balduzzi paper)"""

    def __init__(self, device="cpu"):
        self.strategy_spaces = [Box((1,)), Box((1,))]
    
    def payoffs(self, strategy_a, strategy_b):
        payoff = strategy_a[0] * strategy_b[0]
        return payoff, -payoff