"""Utility classes for defining differentiable games and running experiments"""
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


class Box:  # Do we ever use this anywhere?

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
