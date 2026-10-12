# moving-average

A small Python library for simple (SMA) and exponential (EMA) moving averages with explicit warm-up handling. Standard library only — no dependencies.

## Usage

```python
from moving_average import SimpleMovingAverage, ExponentialMovingAverage

sma = SimpleMovingAverage(period=3)
for v in [1.0, 2.0, 3.0, 4.0]:
    print(sma.add(v))  # None, None, 2.0, 3.0

ema = ExponentialMovingAverage(period=3)
for v in [1.0, 2.0, 3.0, 4.0]:
    print(ema.add(v))  # None, None, 2.0, 2.5
```

Both classes expose `add(value) -> float | None` and an `is_ready` property. `add` returns `None` while fewer than `period` observations have been fed; the call that completes warm-up returns the first real average.

## Why this exists

The awkward part of moving averages is the warm-up window: what do you return before you have enough data? This library makes one concrete decision — during warm-up, `add` returns `None`, and the first average is produced exactly when the `period`-th observation arrives. You handle `None` in your own code; the library does not invent a synthetic pre-fill value.

For the EMA, the seed is the simple mean of the first `period` observations (an "SMA seed"). The alternative — seeding the EMA with the very first sample — makes the early curve depend on a single noisy value. The SMA seed trades a slightly later effective start for a stable, comparable origin. The smoothing coefficient uses the standard `alpha = 2 / (period + 1)`.

## Edge cases

- `period` must be a positive integer; constructing with a smaller value raises `ValueError` (or `TypeError` for non-ints).
- `add` accepts `int` or `float`; anything else raises `TypeError`. Returned averages are always `float`.
- The SMA window is a fixed sliding window; old values are dropped once the window is full.
- The EMA, once seeded, carries information from all subsequent observations (infinite impulse response); it does not forget the seed, but the contribution decays geometrically.

## Project layout

- `src/moving_average/core.py` — `MovingAverage` (base), `SimpleMovingAverage`, `ExponentialMovingAverage`.
- `tests/test_core.py` — `python -m unittest discover -s tests` with `PYTHONPATH=src`.

## Performance

The window keeps a bounded buffer, so `push` is constant time and memory does not
grow with the length of the stream. `peak` and `trough` are linear in the window
size, which is the trade that keeps `push` cheap.

