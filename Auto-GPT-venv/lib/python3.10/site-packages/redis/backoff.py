import random
from abc import ABC, abstractmethod

# Maximum backoff between each retry in seconds
DEFAULT_CAP = 0.512
# Minimum backoff between each retry in seconds
DEFAULT_BASE = 0.008


class AbstractBackoff(ABC):
    """Backoff interface"""

    def reset(self):
        """
        Reset internal state before an operation.
        `reset` is called once at the beginning of
        every call to `Retry.call_with_retry`
        """
        pass

    @abstractmethod
    def compute(self, failures):
        """Compute backoff in seconds upon failure"""
        pass


class ConstantBackoff(AbstractBackoff):
    """Constant backoff upon failure"""

    def __init__(self, backoff):
        """`backoff`: backoff time in seconds"""
        self._backoff = backoff

    def compute(self, failures):
        return self._backoff


class NoBackoff(ConstantBackoff):
    """No backoff upon failure"""

    def __init__(self):
        super().__init__(0)


class ExponentialBackoff(AbstractBackoff):
    """Exponential backoff upon failure"""

    def __init__(self, cap=DEFAULT_CAP, base=DEFAULT_BASE):
        """
        `cap`: maximum backoff time in seconds
        `base`: base backoff time in seconds
        """
        self._cap = cap
        self._base = base

    def compute(self, failures):
        return min(self._cap, self._base * 2**failures)


class FullJitterBackoff(AbstractBackoff):
    """Full jitter backoff upon failure"""

    def __init__(self, cap=DEFAULT_CAP, base=DEFAULT_BASE):
        """
        `cap`: maximum backoff time in seconds
        `base`: base backoff time in seconds
        """
        self._cap = cap
        self._base = base

    def compute(self, failures):
        return random.uniform(0, min(self._cap, self._base * 2**failures))


class EqualJitterBackoff(AbstractBackoff):
    """Equal jitter backoff upon failure"""

    def __init__(self, cap=DEFAULT_CAP, base=DEFAULT_BASE):
        """
        `cap`: maximum backoff time in seconds
        `base`: base backoff time in seconds
        """
        self._cap = cap
        self._base = base

    def compute(self, failures):
        temp = min(self._cap, self._base * 2**failures) / 2
        return temp + random.uniform(0, temp)


class DecorrelatedJitterBackoff(AbstractBackoff):
    """Decorrelated jitter backoff upon failure"""

    def __init__(self, cap=DEFAULT_CAP, base=DEFAULT_BASE):
        """
        `cap`: maximum backoff time in seconds
        `base`: base backoff time in seconds
        """
        self._cap = cap
        self._base = base
        self._previous_backoff = 0

    def reset(self):
        self._previous_backoff = 0

    def compute(self, failures):
        max_backoff = max(self._base, self._previous_backoff * 3)
        temp = random.uniform(self._base, max_backoff)
        self._previous_backoff = min(self._cap, temp)
        return self._previous_backoff


def default_backoff():
    return EqualJitterBackoff()
