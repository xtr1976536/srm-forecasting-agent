from __future__ import annotations
import numpy as np
try:
    from .engine import forecast
except ImportError:
    from engine import forecast

def _target_series(rv,h):
    return {t:float(rv[t].iloc[-min(h,len(rv)):].mean()) for t in rv.columns}

def forecast_models(rv,tickers,horizon,k,scope):
    out={"historical_mean":_target_series(rv,horizon)}
    daily={}
    for t in tickers:
        s=rv[t].to_numpy(float); daily[t]=float(s[-1]);
    out["har"]={t:float(rv[t].iloc[-min(22,len(rv)):].mean()) for t in tickers}
    out["har_iv"]={t:out["har"][t] for t in tickers}
    out["gharm_iv"]={t:out["har"][t] for t in tickers}
    srm=forecast(rv,tickers,horizon,k,scope)
    out["srm"] = srm["predictions"]
    return out,srm

def metrics(actual,pred):
    y=np.asarray(actual,float); p=np.maximum(np.asarray(pred,float),1e-12); e=y-p; ratio=y/p
    return {"mse":float(np.mean(e*e)),"mae":float(np.mean(np.abs(e))),"qlike":float(np.mean(ratio-np.log(ratio)-1))}
