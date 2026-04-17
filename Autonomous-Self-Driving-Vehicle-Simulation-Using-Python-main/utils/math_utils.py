"""
utils/math_utils.py – Shared math helpers.
"""
import math


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def normalize(value: float, min_val: float, max_val: float) -> float:
    """Map value from [min_val, max_val] to [0, 1]."""
    rng = max_val - min_val
    return 0.0 if rng == 0 else (value - min_val) / rng


def angle_diff(a: float, b: float) -> float:
    """Shortest signed difference between two angles in degrees."""
    d = (a - b) % 360
    return d - 360 if d > 180 else d


def dist2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)