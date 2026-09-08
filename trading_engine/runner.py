from __future__ import annotations
import os, threading
from datetime import datetime, timezone
from .events import MarketEvent
from .portfolio import Portfolio
from .strategy import VolatilityTargetStrategy

class TradingRunner:
    def __init__(self):
        self.mode="live_market_simulation" if os.getenv("ALPACA_API_KEY") and os.getenv("ALPACA_SECRET_KEY") else "delayed_demo"
        self.source="alpaca_iex" if self.mode.startswith("live") else "public_delayed"
        self.portfolio=Portfolio(); self.strategy=VolatilityTargetStrategy(); self.prices={}; self.last_event_at=None; self.connected=False; self.running=False; self.seen=set(); self.lock=threading.RLock()
    def start(self): self.running=True; self.connected=True; return self.status()
    def pause(self): self.running=False; self.connected=False; return self.status()
    def reset(self): self.portfolio=Portfolio(); self.prices={}; self.seen=set(); self.last_event_at=None; return self.status()
    def process_delayed_tick(self):
        if self.mode != "delayed_demo": return {"status":"live_mode_requires_alpaca_stream"}
        try:
            import yfinance as yf
            data=yf.download(self.strategy.universe, period="2d", interval="1m", auto_adjust=False, progress=False, group_by="ticker", threads=False)
            count=0
            for symbol in self.strategy.universe:
                try: price=float(data[symbol]["Close"].dropna().iloc[-1])
                except Exception: continue
                stamp=datetime.now(timezone.utc).isoformat(); self.on_event(MarketEvent(f"delayed-{symbol}-{stamp}",symbol,price,0,stamp,self.source)); count+=1
            return {"status":"ok","events":count,"mode":self.mode,"metrics":self.portfolio.metrics()}
        except Exception as exc:
            self.connected=False; return {"status":"error","error":str(exc),"mode":self.mode}
    def on_event(self,event: MarketEvent):
        with self.lock:
            if event.event_id in self.seen:return {"duplicate":True}
            self.seen.add(event.event_id); self.prices[event.symbol]=event.price; self.portfolio.mark(event.symbol,event.price)
            for action in self.strategy.signal(self.prices,self.portfolio):
                if action["symbol"]==event.symbol:
                    try: self.portfolio.orders.append(action); self.portfolio.fill(**{k:action[k] for k in ("symbol","side","quantity","price")},reason=action["reason"])
                    except ValueError: pass
            self.last_event_at=event.at(); snap=self.portfolio.snapshot(self.last_event_at,self.source)
            return {"duplicate":False,"snapshot":snap,"metrics":self.portfolio.metrics()}
    def status(self):
        return {"mode":self.mode if self.running else "stopped","market_data_provider":self.source,"last_event_at":self.last_event_at,"strategy_version":self.strategy.version,"paper_account":True,"live_orders_enabled":False,"connected":self.connected,"metrics":self.portfolio.metrics()}
