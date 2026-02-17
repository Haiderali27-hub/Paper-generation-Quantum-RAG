"""
Math engine: compute latency, speed score, sigmoid probability, final score
"""
import math
from typing import Dict, Any

# Default weights - user can tune
WEIGHTS = {
    'w1': 1.0,  # volume
    'w2': 1.0,  # speed
    'w3': 1.0,  # liquidity
    'w4': 1.0,  # holders growth
    'w5': 1.0,  # anti-rug
    'w6': 1.0   # latency penalty
}


def sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def normalize(value: float, min_v: float, max_v: float) -> float:
    if max_v - min_v == 0:
        return 0.0
    return (value - min_v) / (max_v - min_v)


def latency_score(latency_seconds: float) -> float:
    # lower latency -> higher score; map 0-5s to 1.0-0.0
    return max(0.0, 1.0 - (latency_seconds / 5.0))


def compute_speed_score(metrics: Dict[str, Any]) -> float:
    # metrics expected to have 'volume_change_pct', 'holders_change_pct', 'buys_per_min'
    v = metrics.get('volume_change_pct', 0.0)
    h = metrics.get('holders_change_pct', 0.0)
    b = metrics.get('buys_per_min', 0.0)
    # simple weighted combination
    score = 0.5 * min(max(v, -1.0), 10.0) + 0.3 * min(max(h, -1.0), 10.0) + 0.2 * min(max(b, 0.0), 100.0)
    # normalize roughly to 0-1 range by sigmoid
    return sigmoid(score / 3.0)


def compute_probability(metrics: Dict[str, Any], weights=WEIGHTS) -> float:
    # take normalized inputs or raw where appropriate
    vol = metrics.get('volume_score', 0.0)
    speed = metrics.get('speed_score', 0.0)
    liq = metrics.get('liquidity_score', 0.0)
    hg = metrics.get('holder_growth_score', 0.0)
    anti_rug = metrics.get('anti_rug_score', 1.0)
    latency = metrics.get('latency_seconds', 0.0)

    x = (
        weights['w1'] * vol +
        weights['w2'] * speed +
        weights['w3'] * liq +
        weights['w4'] * hg +
        weights['w5'] * anti_rug -
        weights['w6'] * latency
    )
    return float(sigmoid(x))


def final_score_combination(probability: float, rug_score: float) -> float:
    # rug_score: 0 (safe) - 100 (risky)
    safety = 1.0 - (rug_score / 100.0)
    return probability * safety
