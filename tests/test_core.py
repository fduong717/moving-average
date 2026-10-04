import math
import unittest

from moving_average import MovingAverage, SimpleMovingAverage, ExponentialMovingAverage


def approx_equal(a, b, tol=1e-9):
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


class TestSimpleMovingAverage(unittest.TestCase):

    def test_warmup_returns_none(self):
        sma = SimpleMovingAverage(period=3)
        self.assertIsNone(sma.add(1.0))
        self.assertIsNone(sma.add(2.0))
        self.assertEqual(sma.add(3.0), 2.0)

    def test_is_ready_tracks_warmup(self):
        sma = SimpleMovingAverage(period=3)
        self.assertFalse(sma.is_ready)
        sma.add(1.0)
        sma.add(2.0)
        self.assertFalse(sma.is_ready)
        sma.add(3.0)
        self.assertTrue(sma.is_ready)

    def test_window_slides(self):
        sma = SimpleMovingAverage(period=3)
        sma.add(1.0)
        sma.add(2.0)
        sma.add(3.0)  # mean 2.0
        self.assertEqual(sma.add(4.0), 3.0)  # {2,3,4}
        self.assertEqual(sma.add(5.0), 4.0)  # {3,4,5}

    def test_period_one(self):
        sma = SimpleMovingAverage(period=1)
        self.assertEqual(sma.add(7.0), 7.0)
        self.assertEqual(sma.add(3.0), 3.0)

    def test_accepts_int_input(self):
        sma = SimpleMovingAverage(period=2)
        self.assertIsNone(sma.add(1))
        self.assertEqual(sma.add(3), 2.0)
        self.assertIsInstance(sma.add(5), float)

    def test_negative_values(self):
        sma = SimpleMovingAverage(period=2)
        sma.add(-1.0)
        sma.add(1.0)
        self.assertEqual(sma.add(-3.0), -1.0)

    def test_invalid_period(self):
        with self.assertRaises(ValueError):
            SimpleMovingAverage(period=0)
        with self.assertRaises(ValueError):
            SimpleMovingAverage(period=-2)

    def test_non_numeric_value(self):
        sma = SimpleMovingAverage(period=2)
        with self.assertRaises(TypeError):
            sma.add("x")


class TestExponentialMovingAverage(unittest.TestCase):

    def test_warmup_returns_none(self):
        ema = ExponentialMovingAverage(period=3)
        self.assertIsNone(ema.add(1.0))
        self.assertIsNone(ema.add(2.0))
        self.assertEqual(ema.add(3.0), 2.0)  # seed = mean of 1,2,3

    def test_is_ready_after_warmup(self):
        ema = ExponentialMovingAverage(period=3)
        self.assertFalse(ema.is_ready)
        ema.add(1.0)
        ema.add(2.0)
        self.assertFalse(ema.is_ready)
        ema.add(3.0)
        self.assertTrue(ema.is_ready)

    def test_seed_is_simple_mean(self):
        ema = ExponentialMovingAverage(period=4)
        for v in [1.0, 2.0, 3.0, 4.0]:
            result = ema.add(v)
        self.assertEqual(result, 2.5)

    def test_recurrence_after_seed(self):
        period = 3
        alpha = 2.0 / (period + 1)
        ema = ExponentialMovingAverage(period=period)
        # Fill seed with ones -> ema = 1.0
        for _ in range(period):
            ema.add(1.0)
        # Now feed a single 0.0; standard EMA recurrence
        result = ema.add(0.0)
        expected = alpha * 0.0 + (1.0 - alpha) * 1.0
        self.assertTrue(approx_equal(result, expected))

    def test_steady_state_converges_to_input(self):
        ema = ExponentialMovingAverage(period=5)
        for _ in range(200):
            ema.add(42.0)
        last = ema.add(42.0)
        self.assertTrue(approx_equal(last, 42.0))

    def test_period_one_emits_each_value(self):
        # period=1: seed is the first value, then EMA recurrence with alpha=1.0
        ema = ExponentialMovingAverage(period=1)
        self.assertEqual(ema.add(5.0), 5.0)
        self.assertEqual(ema.add(7.0), 7.0)
        self.assertEqual(ema.add(3.0), 3.0)

    def test_negative_values(self):
        ema = ExponentialMovingAverage(period=2)
        ema.add(-1.0)
        ema.add(1.0)  # seed = mean(-1, 1) = 0.0
        result = ema.add(-1.0)
        alpha = 2.0 / 3.0
        expected = alpha * (-1.0) + (1.0 - alpha) * 0.0
        self.assertTrue(approx_equal(result, expected))

    def test_invalid_period(self):
        with self.assertRaises(ValueError):
            ExponentialMovingAverage(period=0)
        with self.assertRaises(TypeError):
            ExponentialMovingAverage(period=2.5)

    def test_non_numeric_value(self):
        ema = ExponentialMovingAverage(period=2)
        with self.assertRaises(TypeError):
            ema.add(None)


class TestMovingAverageBase(unittest.TestCase):

    def test_period_property(self):
        sma = SimpleMovingAverage(period=4)
        self.assertEqual(sma.period, 4)
        ema = ExponentialMovingAverage(period=4)
        self.assertEqual(ema.period, 4)

    def test_base_is_abstract(self):
        ma = MovingAverage(period=2)
        with self.assertRaises(NotImplementedError):
            ma.add(1.0)


if __name__ == "__main__":
    unittest.main()
