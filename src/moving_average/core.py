from __future__ import annotations

from collections import deque
from typing import Optional


class MovingAverage:
    """Base class for moving average implementations.

    Each concrete subclass returns either a float value or None while
    warming up (fewer than `period` observations have been fed in).

    `period` is the number of observations required to consider the
    average "ready". For the SMA this doubles as the window size; for
    the EMA it only governs warm-up, since EMAs by definition carry
    information from all prior observations.
    """

    __slots__ = ("_period",)

    def __init__(self, period: int) -> None:
        if not isinstance(period, int):
            raise TypeError("period must be an int")
        if period < 1:
            raise ValueError("period must be >= 1")
        self._period = period

    @property
    def period(self) -> int:
        return self._period

    @property
    def is_ready(self) -> bool:
        """True once at least `period` observations have been fed."""
        return len(self._observations()) >= self._period

    def add(self, value: float) -> Optional[float]:
        """Record a new observation and return the updated average, or None while warming up."""
        raise NotImplementedError

    def _observations(self):
        raise NotImplementedError


class SimpleMovingAverage(MovingAverage):
    """Fixed-window simple moving average.

    A deque holds the last `period` observations. On each add we push
    the new value, drop the oldest if the window is full, and return
    the arithmetic mean of the window. Returns None until the window
    fills for the first time.
    """

    __slots__ = ("_window", "_sum")

    def __init__(self, period: int) -> None:
        super().__init__(period)
        self._window: deque = deque(maxlen=period)
        self._sum: float = 0.0

    def add(self, value: float) -> Optional[float]:
        if not isinstance(value, (int, float)):
            raise TypeError("value must be numeric")
        v = float(value)
        if len(self._window) == self._period:
            # deque(maxlen=...) already evicts the oldest on append,
            # so subtract before appending to keep the sum correct.
            self._sum -= self._window[0]
        self._window.append(v)
        self._sum += v
        if len(self._window) < self._period:
            return None
        return self._sum / len(self._window)

    def _observations(self):
        return self._window


class ExponentialMovingAverage(MovingAverage):
    """Exponential moving average with explicit warm-up.

    Design decision: the first `period` observations are averaged
    to seed the EMA (a "SMA seed"). Only once the seed is established
    does exponential weighting begin. This is a common, well-defined
    choice: pure recursive seeding (EMA = first value) makes the
    early average depend heavily on a single sample, while an SMA
    seed gives the warm-up curve a stable, comparable starting point.

    The smoothing coefficient follows the standard convention
    alpha = 2 / (period + 1).

    During warm-up (before `period` values have arrived) `add`
    returns None; internally it accumulates the seed sum. The first
    call after warm-up completes the seed and returns the seeded EMA;
    subsequent calls apply the standard EMA recurrence.
    """

    __slots__ = ("_alpha", "_seed_count", "_seed_sum", "_ema")

    def __init__(self, period: int) -> None:
        super().__init__(period)
        self._alpha: float = 2.0 / (period + 1)
        self._seed_count: int = 0
        self._seed_sum: float = 0.0
        self._ema: Optional[float] = None

    def add(self, value: float) -> Optional[float]:
        if not isinstance(value, (int, float)):
            raise TypeError("value must be numeric")
        v = float(value)
        if self._seed_count < self._period:
            self._seed_sum += v
            self._seed_count += 1
            if self._seed_count == self._period:
                self._ema = self._seed_sum / self._period
                return self._ema
            return None
        self._ema = self._alpha * v + (1.0 - self._alpha) * self._ema
        return self._ema

    def _observations(self):
        # Used only for is_ready; we report the seed count.
        class _Counter:
            def __len__(self_inner):
                return self._seed_count
        return _Counter()
