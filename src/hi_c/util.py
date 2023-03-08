"""Utility classes for defining differentiable games"""
import math
from numbers import Real

class Box:

    def __init__(self, min, max, shape):
        self.min = min
        self.max = max
        self.shape = shape

    def sample(self, rng):
        return (self.max - self.min) * rng.random(self.shape) + self.min


class ReversedGame:
    """Wraps an existing differentiable game with the player IDs reversed"""
    
    def __init__(self, game):
        self._game = game
        self.strategy_spaces = [
            game.strategy_spaces[1],
            game.strategy_spaces[0],
        ]

    def payoffs(self, strategy_a, strategy_b):
        payoff_a, payoff_b = self._game.payoffs(strategy_b, strategy_a)
        return payoff_b, payoff_a
    

class FixedSchedule:

    def __init__(self, value):
        self._value = value
    
    def step(self):
        return self._value


class PSeriesSchedule:

    def __init__(self, scale, exponent):
        self._scale = scale
        self._exponent = -exponent
        self._step = 0
    
    def step(self):
        self._step += 1
        return self._scale * self._step ** self._exponent


class LogSchedule:

    def __init__(self, scale):
        self._scale = scale
        self._step = 0

    def step(self):
        self._step += 1
        return self._scale * math.log(self._step)


def get_schedule(value):
    if isinstance(value, dict):
        name, params = list(value.items())[0]

        if "fixed" == name:
            if isinstance(params, Real):
                return FixedSchedule(params)
            else:
                return FixedSchedule(**params)
        elif "p_series" == name:
            return PSeriesSchedule(**params)
        elif "logarithmic" == name:
            return LogSchedule(**params)
        else:
            raise ValueError(f"No schedule '{name}' is defined")
