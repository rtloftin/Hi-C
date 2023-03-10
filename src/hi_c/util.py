"""Utility classes for defining differentiable games"""
import math
from numbers import Real
import numpy as np
import time

class Stopwatch:

    def __init__(self):
        self._started = None
        self._latest = 0
        self._elapsed = 0
    
    def start(self):
        if self._started is None:
            self._started = time.time()

    def stop(self):
        stopped = time.time()
        if self._started is not None:
            self._latest = stopped - self._started
            self._elapsed += self._latest
            self._started = None
        else:
            raise Exception("Tried to stop a stopwatch that had not been started")

    @property
    def latest(self):
        return self._latest

    @property
    def elapsed(self):
        return self._elapsed


class Box:

    def __init__(self, shape, min=-np.inf, max=np.inf):
        self.min = min
        self.max = max
        self.shape = shape


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

        if "p_series" == name:
            return PSeriesSchedule(**params)
        elif "logarithmic" == name:
            return LogSchedule(**params)
        else:
            raise ValueError(f"No schedule '{name}' is defined")
    elif isinstance(value, Real):
        return FixedSchedule(value)
    else:
        raise ValueError("Parameter schedule must either be a real value or a dictionary")
