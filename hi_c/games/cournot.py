from hi_c.util import Box


class Cournot:
    """
    The Cournot competition with two players.  The price function and individual cost functions are both linear.

    Payoff functions:
        f_1(q_1, q_2) = q_1 [P_0 - S (q_1 + q_2)] - C_1 q_1
        f_2(q_1, q_2) = q_2 [P_0 - S (q_1 + q_2)] - C_2 q_2

    Where:
        S - price slope
        P_0 - initial price
        C_1 - player 1's unit cost
        C_2 - player 2's unit cost
    """

    def __init__(self,
                 initial_price=50,
                 price_slope=1,
                 cost_1=1,
                 cost_2=1,
                 init_range=50.,
                 device="cpu"):
        self.initial_price = initial_price
        self.price_slope = price_slope
        self.cost_1 = cost_1
        self.cost_2 = cost_2

        self.strategy_spaces = [
            Box(0., init_range, (1, )),
            Box(0., init_range, (1, ))
        ]

    def payoffs(self, quantity_1, quantity_2):
        price = self.initial_price - self.price_slope * (quantity_1[0] + quantity_2[0])
        payoff_1 = quantity_1[0] * (price - self.cost_1)
        payoff_2 = quantity_2[0] * (price - self.cost_2)

        return payoff_1, payoff_2

    @property
    def stackelberg_equilibrium(self):
        """Returns the Stackelberg equilibrium of the game with player 1 as the market leader"""
        leader = (0.5 * (self.initial_price + self.cost_2) - self.cost_1) / self.price_slope
        follower = 0.5 * (self.initial_price - self.price_slope * leader - self.cost_2) / self.price_slope

        return leader, follower
