from trading_engine.events import MarketEvent
from trading_engine.runner import TradingRunner

def test_delayed_runner_updates_account_and_deduplicates():
    r=TradingRunner(); r.start(); event=MarketEvent("e1","AAPL",100,10,"2026-01-01T14:30:00Z")
    first=r.on_event(event); second=r.on_event(event)
    assert first["duplicate"] is False and second["duplicate"] is True
    assert r.status()["metrics"]["nav"] > 0

def test_runner_never_enables_live_orders():
    assert TradingRunner().status()["live_orders_enabled"] is False

def test_strategy_creates_simulated_fill():
    r=TradingRunner(); r.start(); out=r.on_event(MarketEvent("e1","AAPL",100,10,"2026-01-01T14:30:00Z"))
    assert out["metrics"]["fills"] >= 1 and out["metrics"]["fees"] > 0
