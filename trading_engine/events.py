from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone

@dataclass(frozen=True)
class MarketEvent:
    event_id: str
    symbol: str
    price: float
    volume: float = 0.0
    timestamp: str = ""
    source: str = "public_delayed"
    def at(self): return self.timestamp or datetime.now(timezone.utc).isoformat()
