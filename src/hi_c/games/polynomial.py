from hi_c.util import Box

class TandemGame:
    """The tandem bicycle game from the SOS paper (need to find the reference for this)"""

    def __init__(self, config={}):
        range = config.get("range", 100)
        self.strategy_spaces = [
            Box(-range, range, (1,)),
            Box(-range, range, (1,))
        ]
    
    def payoffs(self, strategy_a, strategy_b):
        base = strategy_a + strategy_b
        base = base * base

        return base - 2 * strategy_a, base - 2 * strategy_b


class HamiltonianGame:
    """The Hamiltonian game (referenced in the SOS paper, but orgininally from a Balduzzi paper)"""

    def __init__(self, config={}):
        range = config.get("range", 100)
        self.strategy_spaces = [
            Box(-range, range, (1,)),
            Box(-range, range, (1,))
        ]
    
    def payoffs(self, strategy_a, strategy_b):
        payoff = strategy_a * strategy_b
        return payoff, -payoff