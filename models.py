from __future__ import annotations
import numpy as np
from engine import forecast

def _har_forecast(series: np.ndarray, horizon: int) -> float:
    """Fit a rolling HAR OLS using only observations before the forecast origin."""
    n = len(series)
    origins = np.arange(22, n - horizon + 1)
    if len(origins) < 30:
        return float(series[-1])
    x, y = [], []
    for t in origins:
        x.append([1.0, series[t-1], series[t-5:t-1].mean(), series[t-22:t-5].mean()])
        y.append(series[t:t+horizon].mean())
    coef, *_ = np.linalg.lstsq(np.asarray(x), np.asarray(y), rcond=None)
    current = np.asarray([1.0, series[-1], series[-5:-1].mean(), series[-22:-5].mean()])
    return float(max(current @ coef, 1e-12))

def forecast_models(rv, tickers, horizon, k, scope):
    out = {
        "historical_mean": {t: float(rv[t].iloc[-min(horizon, len(rv)):].mean()) for t in tickers},
        "har": {t: _har_forecast(rv[t].to_numpy(float), horizon) for t in tickers},
        "har_iv": None,
        "gharm": None,
        "gharm_iv": None,
    }
    srm = forecast(rv, tickers, horizon, k, scope)
    out["srm"] = srm["predictions"]
    return out, srm

def metrics(actual, pred):
    y=np.asarray(actual,float); p=np.maximum(np.asarray(pred,float),1e-12); e=y-p; ratio=y/p
    return {"mse":float(np.mean(e*e)),"mae":float(np.mean(np.abs(e))),"qlike":float(np.mean(ratio-np.log(ratio)-1))}
