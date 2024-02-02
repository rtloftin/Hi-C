# This script just runs experiments with Hi-C in the Cournot game

# TODO: Move this into the main codebase

import math
import matplotlib.pyplot as plot
import numpy as np

from hi_c.util import Box
from hi_c.learners.gradient import NaiveLearner
from hi_c.learners.hi_c import HiC
from hi_c.learners.hierarchical import HierarchicalGradient
from hi_c.learners.schedule import FixedSchedule, LogSchedule, PSeriesSchedule


# This class appears to describe the cournot game with linear costs
class CournotLinear:

    def __init__(self,
                 initial_price=50,
                 price_slope=1,
                 cost_1=1,
                 cost_2=1):
        self.initial_price = initial_price
        self.price_slope = price_slope
        self.cost_1 = cost_1
        self.cost_2 = cost_2

        # Have to define a strategy space for each player - in this case a scalar production volume
        self.strategy_spaces = (
            Box((1, ), 0., 1000.),
            Box((1, ), 0., 1000.)
        )

    def payoffs(self, quantity_1, quantity_2):
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


class TandemGame:  # I Don't think we had results for this
    """The tandem bicycle game from the SOS paper (need to find the reference for this)"""

    def __init__(self):
        self.strategy_spaces = [Box((1,)), Box((1,))]

    def payoffs(self, params_a, params_b):
        base = (params_a[0] + params_b[0]) ** 2
        return base - 2 * params_a[0], base - 2 * params_b[0]

    @property
    def equilibrium(self):
        return 0., 0.  # NOTE: This is NOT necessarily the Stackelberg solution


if __name__ == '__main__':  # TODO: Move this to a YAML configuraation
    ITERATIONS = 1000000  # NOTE: This appears to be the number of "inner" iterations
    REPORT = 25000
    THRESHOLD = 297.  # NOTE: This allows us to compute the time required to reach the Stackelberg equilibrium payoff

    game = CournotLinear()
    # game = TandemGame()

    # NOTE: These are altogether different experiments
    # learner_1 = NaiveLearner(game, 0, lr=PSeriesSchedule(0.01, .55))
    # learner_2 = NaiveLearner(game, 1, lr=PSeriesSchedule(0.01, .55))

    # learner_1 = HierarchicalGradient(game, 0, lr=PSeriesSchedule(0.1, .001))
    # learner_2 = NaiveLearner(game, 1, lr=PSeriesSchedule(0.01, .525))
    # learner_2 = NaiveLearner(game, 1, lr=FixedSchedule(0.1))

    print("\nInitializing Hi-C learner:")

    # Definitely something wrong here
    # lr_exponent = 0.001
    lr_exponent = 0.1
    # perturbation_exponent = 0.5
    perturbation_exponent = 0.6
    inner_lr = 0.1
    
    B = 2. * game.initial_price
    # B = 50  # NOTE: the tandem game doesn't define the "price slope"
    z = math.log(1. - inner_lr * 2. * game.price_slope)
    # z = math.log(1. - inner_lr * 2.)  # NOTE: again, the tandem game doesn't define the necessary parameters
    commitment_scale = -2. * (perturbation_exponent + 1.) / z
    commitment_offset = -2. * math.log(B) / z
    
    print(f"    outer learning rate exponent: {lr_exponent}")
    print(f"    perturbation schedule exponent: {perturbation_exponent}")
    print(f"    fixed inner learning rate: {inner_lr}")
    print(f"    commitment schedule scale: {commitment_scale}")
    print(f"    commitment schedule offset: {commitment_offset}")
    
    learner_1 = HiC(game, 
                    0, 
                    # lr=PSeriesSchedule(0.00001, lr_exponent),  # NOTE: Why is the learning rate so small?
                    lr=PSeriesSchedule(0.001, lr_exponent),
                    p=PSeriesSchedule(1., perturbation_exponent),
                    k=LogSchedule(commitment_scale, commitment_offset),
                    baseline_lambda=0.9,
                    burn_in=50)
    learner_2 = NaiveLearner(game, 1, lr=FixedSchedule(inner_lr))

    strategies = [learner_1.reset(), learner_2.reset()]
    print("\nBEGIN TRAINING\n\n")
    print(f"Equilibrium strategy: {game.equilibrium}")
    print(f"Initial strategies: {strategies}")
    print(f"Initial payoffs: {game.payoffs(*strategies)}")

    joint = [s.detach().cpu().numpy() for s in strategies]
    leader_payoffs = [game.payoffs(*joint)[0]]
    equilibrium_errors = [sum((target - value)**2 for target, value in zip(game.equilibrium, joint))]

    threshold_point = np.inf

    for iteration in range(ITERATIONS):
        strategies = [learner_1.step(strategies), learner_2.step(strategies)]

        # Compute metrics
        joint = [s.detach().cpu().numpy() for s in strategies]
        leader_payoffs.append(game.payoffs(*joint)[0])
        equilibrium_errors.append(sum((target - value) ** 2 for target, value in zip(game.equilibrium, joint)))

        if threshold_point == np.inf and leader_payoffs[-1] >= THRESHOLD:
            threshold_point = iteration + 1

        if (iteration + 1) % REPORT == 0:
            print(f"\nIteration {iteration + 1}:")
            print(f"    strategies: {strategies}")
            print(f"    payoffs: {game.payoffs(*strategies)}\n")

    print(f"COMPLETE - reached leader payoff threshold in {threshold_point} iterations")

    # NOTE: Already have plotting code
    x_axis = np.arange(ITERATIONS + 1)
    plot.switch_backend("qtagg")
    plot.clf()
    plot.plot(x_axis, leader_payoffs)  # NOTE: Plotting leader payoffs
    plot.show()
