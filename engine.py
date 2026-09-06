from dataclasses import dataclass
import numpy as np

@dataclass
class Neighbor:
    ticker:str; endpoint:int; distance:float; relative_change:float

def curve(series,endpoint,length=22):
    x=np.log(np.maximum(series,1e-8)); p=x[endpoint-length+1:endpoint+1]
    if len(p)!=length: raise ValueError("insufficient history for path")
    z=(p-p.mean())/(p.std()+1e-8)
    return np.asarray([(z[r],z[max(0,r-1)],z[max(0,r-2)]) for r in range(length)],float)

def forecast(rv,tickers,horizon=1,k=20):
    endpoint=len(rv)-1; candidates=[]
    for ticker in tickers:
        series=rv[ticker].to_numpy(float); query=curve(series,endpoint)
        for s in range(21,endpoint-horizon+1):
            d=float(np.sqrt(np.mean((query-curve(series,s))**2))); future=float(series[s+1:s+horizon+1].mean()); rel=float(np.log(future/(series[s]+1e-8)))
            candidates.append(Neighbor(ticker,s,d,rel))
    if not candidates: raise ValueError("no historical candidates")
    candidates.sort(key=lambda n:n.distance); neighbors=candidates[:min(k,len(candidates))]; d=np.asarray([n.distance for n in neighbors]); w=np.exp(-(d-d.min())); w/=w.sum(); change=float(w@np.asarray([n.relative_change for n in neighbors]))
    return {"forecast_date":str(rv.index[-1].date()),"horizon":horizon,"k":len(neighbors),"predictions":{t:float(rv[t].iloc[-1]*np.exp(change)) for t in tickers},"neighbors":[n.__dict__ for n in neighbors],"weights":w.tolist(),"candidate_count":len(candidates)}
