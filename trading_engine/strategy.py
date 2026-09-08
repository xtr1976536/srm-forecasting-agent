from __future__ import annotations
class VolatilityTargetStrategy:
    version="vol-target-v1"
    def __init__(self, universe=("AAPL","MSFT","JPM"), max_weight=.35, target_vol=.10): self.universe=list(universe); self.max_weight=max_weight; self.target_vol=target_vol
    def signal(self, prices, portfolio):
        available=[s for s in self.universe if s in prices and prices[s]>0]
        if not available:return []
        weight=min(self.max_weight, 1/max(len(available),1))
        target={s:portfolio.nav()*weight for s in available}; actions=[]
        for symbol,value in target.items():
            current=portfolio.positions.get(symbol,0)*prices[symbol]
            delta=value-current
            if abs(delta)>max(25, portfolio.nav()*.005): actions.append({"symbol":symbol,"side":"buy" if delta>0 else "sell","quantity":abs(delta)/prices[symbol],"price":prices[symbol],"reason":self.version})
        return actions
