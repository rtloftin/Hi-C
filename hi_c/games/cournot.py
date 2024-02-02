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
                 cost_2=1):
        self.initial_price = initial_price
        self.price_slope = price_slope
        self.cost_1 = cost_1
        self.cost_2 = cost_2


        self.strategy_spaces = (
            Box((1, ), 0., 1000.),
            Box((1, ), 0., 1000.)
        )

    def payoffs(self, quantity_1, quantity_2):  # TODO: Convert these to vectors
        price = self.initial_price - self.price_slope * (quantity_1 + quantity_2)  # Price goes down as supply goes up
        payoff_1 = quantity_1 * (price - self.cost_1)  # Payoff is quantity times profit-per-unit
        payoff_2 = quantity_2 * (price - self.cost_2)  # Game may have asymmetric costs

        return payoff_1, payoff_2  # TODO: Make sure we have the sign right for the gradient step - what was this for?

    # Used to compute the "distance" from the unique Stackelberg equilibrium (not obvious that is what this is)
    @property
    def equilibrium(self):
        """
            U(q_2 : q_1) = q_2 P(q_1 + q_2) + C_2(q_2)
                         = q_2 (p_0 - s (q_1 + q_2) - c_2)
                         = q_2 (p_0 - s q_1 - c_2) - s (q_2)^2
            U'(q_2 : q_1) = (p_0 - s q_1 - c_2) - 2 s q_2
                     q_2 =  (p_0 - s q_1 - c_2) / 2 s = r(q_1)
            r'(q_1) = -2
            U(q_1) = q_1 (p_0 - s (q_1 + r(q_1)) - c_1)
            U'(q_1) = (p_0 - c_1) - 2 s q_1 - s (r(q_1) - 2 q_1)
                    = (p_0 - c_1) - (p_0 - s q_1 - c_2) / 2
                    = s q_1 / 2 - c_1 + c_2 / 2 + p_0 / 2
                q_1 = (2 / s) (c_1 - c_2 / 2 - p_0 / 2)
                    = (2 c_1 - c_2 - p_0) / s
        """
        leader = (0.5 * (self.initial_price + self.cost_2) - self.cost_1) / self.price_slope
        follower = 0.5 * (self.initial_price - self.price_slope * leader - self.cost_2) / self.price_slope

        return leader, follower