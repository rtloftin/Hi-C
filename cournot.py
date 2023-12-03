import math

from hi_c.util import Box
from hi_c.learners.gradient import NaiveLearner
from hi_c.learners.hi_c import HiC
from hi_c.learners.hierarchical import HierarchicalGradient
from hi_c.learners.schedule import FixedSchedule, LogSchedule, PSeriesSchedule


class CournotLinear:

    def __init__(self,
                 initial_price=50,
                 price_slope=1.5,
                 cost_1=1,
                 cost_2=.5):
        self.initial_price = initial_price
        self.price_slope = price_slope
        self.cost_1 = cost_1
        self.cost_2 = cost_2

        self.strategy_spaces = (
            Box((1, ), 0., 1000.),
            Box((1, ), 0., 1000.)
        )

    def payoffs(self, quantity_1, quantity_2):
        price = self.initial_price - self.price_slope * (quantity_1 + quantity_2)
        payoff_1 = quantity_1 * (price - self.cost_1)
        payoff_2 = quantity_2 * (price - self.cost_2)

        return payoff_1, payoff_2  # TODO: Make sure we have the sign right for the gradient step

    @property
    def equilibrium(self):
        leader = (0.5 * (self.initial_price + self.cost_2) - self.cost_1) / self.price_slope
        follower = (self.initial_price - self.price_slope * leader - self.cost_2) / (2 * self.price_slope)

        return leader, follower


if __name__ == '__main__':
    ITERATIONS = 5000000
    REPORT = 100000

    game = CournotLinear()

    # learner_1 = NaiveLearner(game, 0, lr=PSeriesSchedule(0.01, .55))
    # learner_2 = NaiveLearner(game, 1, lr=PSeriesSchedule(0.01, .55))

    # learner_1 = HierarchicalGradient(game, 0, lr=PSeriesSchedule(0.01, .55))
    # learner_2 = NaiveLearner(game, 1, lr=PSeriesSchedule(0.01, .525))
    # learner_2 = NaiveLearner(game, 1, lr=FixedSchedule(0.1))

    print("\nInitializing Hi-C learner:")

    # Definitely something wrong here
    lr_exponent = 0.1
    perturbation_exponent = 0.6
    inner_lr = 0.1
    
    B = 2. * game.initial_price
    z = math.log(1. - inner_lr * 2. * game.price_slope)
    commitment_scale = -2. * (perturbation_exponent + 1.) / z
    commitment_offset = -2. * math.log(B) / z
    
    print(f"    outer learning rate exponent: {lr_exponent}")
    print(f"    perturbation schedule exponent: {perturbation_exponent}")
    print(f"    fixed inner learning rate: {inner_lr}")
    print(f"    commitment schedule scale: {commitment_scale}")
    print(f"    commitment schedule offset: {commitment_offset}")
    
    learner_1 = HiC(game, 
                    0, 
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

    for iteration in range(ITERATIONS):
        strategies = [learner_1.step(strategies), learner_2.step(strategies)]

        if (iteration + 1) % REPORT == 0:
            print(f"\nIteration {iteration + 1}:")
            print(f"    strategies: {strategies}")
            print(f"    payoffs: {game.payoffs(*strategies)}\n")

    print("COMPLETE")
