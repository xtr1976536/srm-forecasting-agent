from __future__ import annotations

import numpy as np


def return_scenario(returns, last_price: float, horizon: int = 1, confidence: float = 0.90) -> dict:
    """Independent return scenario model; it is not part of SRM.

    The model uses a rolling drift and volatility estimate only to create a
    transparent price scenario band for paper simulation.
    """
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 20:
        raise ValueError("at least 20 returns are required for a return scenario")
    window = values[-60:]
    drift = float(np.mean(window))
    volatility = float(np.std(window, ddof=1))
    z = 1.6448536269514722 if confidence >= 0.9 else 1.2815515655446004
    mean_return = horizon * drift
    sigma = np.sqrt(max(horizon, 1)) * volatility
    center = float(last_price * np.exp(mean_return))
    lower = float(last_price * np.exp(mean_return - z * sigma))
    upper = float(last_price * np.exp(mean_return + z * sigma))
    direction_probability = float(1.0 / (1.0 + np.exp(-(mean_return / (sigma + 1e-12)))))
    return {
        "horizon": int(horizon),
        "last_price": float(last_price),
        "drift": drift,
        "volatility": volatility,
        "expected_price": center,
        "lower_price": lower,
        "upper_price": upper,
        "probability_positive_return": direction_probability,
        "model": "rolling return scenario",
        "warning": "Scenario model for research simulation; not an investment recommendation.",
    }
