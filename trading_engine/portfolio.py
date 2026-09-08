from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class Portfolio:
    initial_cash: float = 100000.0
    cash: float = 100000.0
    positions: dict[str, float] = field(default_factory=dict)
    marks: dict[str, float] = field(default_factory=dict)
    orders: list[dict] = field(default_factory=list)
    fills: list[dict] = field(default_factory=list)
    equity: list[dict] = field(default_factory=list)
    fees: float = 0.0
    def mark(self, symbol, price): self.marks[symbol]=float(price)
    def nav(self): return self.cash + sum(q*self.marks.get(s,0) for s,q in self.positions.items())
    def fill(self, symbol, side, quantity, price, fee=0.0005, reason="strategy"):
        gross=float(quantity)*float(price); cost=gross*fee
        if side=="buy" and gross+cost>self.cash: raise ValueError("insufficient cash")
        held=self.positions.get(symbol,0.0)
        if side=="sell" and quantity>held: raise ValueError("insufficient position")
        self.cash += -gross-cost if side=="buy" else gross-cost
        self.positions[symbol]=held+(quantity if side=="buy" else -quantity)
        self.fees += cost
        item={"symbol":symbol,"side":side,"quantity":quantity,"price":price,"fee":cost,"reason":reason}
        self.fills.append(item); return item
    def snapshot(self, timestamp, source="public_delayed"):
        nav=self.nav(); self.equity.append({"timestamp":timestamp,"nav":nav,"cash":self.cash,"source":source})
        return self.equity[-1]
    def metrics(self):
        values=[x["nav"] for x in self.equity]; peak=0; max_dd=0
        for value in values:
            peak=max(peak,value); max_dd=min(max_dd, value/peak-1 if peak else 0)
        return {"nav":self.nav(),"cash":self.cash,"fees":self.fees,"return_pct":(self.nav()/self.initial_cash-1)*100,"max_drawdown_pct":max_dd*100,"positions":dict(self.positions),"orders":len(self.orders),"fills":len(self.fills)}
