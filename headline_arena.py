"""Safety-first Headline Arena payload builder.

This module deliberately does not submit network requests.  It validates and
locks a caller-supplied direction probability so a future connector can be
added without changing the SRM engine or weakening its information boundary.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_payload(*, target: str, probability_positive: float,
                  forecast_time: str | None = None,
                  source: str = "srm-forecasting-agent") -> dict[str, Any]:
    """Build a deterministic, JSON-safe dry-run payload.

    ``probability_positive`` must be supplied by a direction model; SRM
    volatility magnitudes must not be converted into direction implicitly.
    """
    if not target or not target.strip():
        raise ValueError("target is required")
    p = float(probability_positive)
    if not 0.0 <= p <= 1.0:
        raise ValueError("probability_positive must be between 0 and 1")
    locked = forecast_time or datetime.now(timezone.utc).isoformat()
    try:
        datetime.fromisoformat(locked.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("forecast_time must be ISO-8601") from exc
    return {
        "target": target.strip(),
        "probability_positive": p,
        "direction": "up" if p >= 0.5 else "down",
        "forecast_time": locked,
        "source": source,
        "mode": "dry-run",
        "network_submission": False,
    }
