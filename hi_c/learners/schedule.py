from abc import ABCMeta, abstractmethod
import math
from numbers import Real


class Schedule(metaclass=ABCMeta):

    @abstractmethod
    def step(self):
        raise NotImplementedError


class FixedSchedule(Schedule):

    def __init__(self, value):
        self._value = value

    def step(self):
        return self._value


class PSeriesSchedule(Schedule):

    def __init__(self, scale, exponent):
        self._scale = scale
        self._exponent = -exponent
        self._step = 0

    def step(self):
        self._step += 1
        return self._scale * self._step ** self._exponent


class LogSchedule(Schedule):

    def __init__(self, scale, offset):
        self._scale = scale
        self._offset = offset
        self._step = 0

    def step(self):
        self._step += 1
        return self._scale * math.log(self._step) + self._offset


def get_schedule(value):
    if isinstance(value, Schedule):
        return value
    elif isinstance(value, Real):
        return FixedSchedule(value)
    elif isinstance(value, dict):
        name, params = list(value.items())[0]

        if "p_series" == name:
            return PSeriesSchedule(**params)
        elif "logarithmic" == name:
            return LogSchedule(**params)
        else:
            raise ValueError(f"No schedule '{name}' is defined")
    else:
        raise ValueError("Parameter schedule must either be a real value, a schedule object, or a config dictionary")
