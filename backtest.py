from __future__ import annotations
import numpy as np
import pandas as pd
from full_srm import run_full_srm
from full_baselines import run_full_baselines
from models import metrics

def run_backtest(panel,tickers,horizon,model="srm",scope="cross_asset",neighbors=20,criterion="qlike",origins=3):
    if model not in {"srm","har","har_iv","gharm","gharm_iv"}: raise ValueError("unsupported backtest model")
    minimum=22+600+horizon+200+horizon
    last=len(panel.rv)-horizon
    first=max(minimum,last-int(origins)+1)
    if first>last: raise ValueError(f"backtest requires at least {minimum+horizon} observations")
    rows=[]
    for t in range(first,last+1):
        rv=panel.rv.iloc[:t]; returns=panel.returns.iloc[:t]; iv=panel.iv.iloc[:t] if panel.iv is not None else None
        if model=="srm": prediction=run_full_srm(rv,tickers,horizon,scope,criterion,neighbors)["predictions"]
        else:
            result=run_full_baselines(rv,iv,returns,tickers,horizon,criterion)
            if model not in result["predictions"]: raise ValueError(f"{model} requires inputs not available in this data mode")
            prediction=result["predictions"][model]
        for ticker in tickers:
            actual=float(panel.rv[ticker].iloc[t:t+horizon].mean())
            rows.append({"Date":str(panel.rv.index[t].date()),"Ticker":ticker,"Model":model,"Target":actual,"Prediction":prediction[ticker],"Error":actual-prediction[ticker]})
    frame=pd.DataFrame(rows); summary=metrics(frame["Target"],frame["Prediction"])
    threshold=float(frame["Target"].quantile(.9)); frame["Regime"]=np.where(frame["Target"]>threshold,"High","Normal")
    regimes={name:metrics(group["Target"],group["Prediction"]) for name,group in frame.groupby("Regime")}
    return {"summary":summary,"regimes":regimes,"predictions":frame,"config":{"horizon":horizon,"model":model,"scope":scope,"neighbors":neighbors,"criterion":criterion,"origins":origins},"audit":{"first_origin":int(first),"last_origin":int(last),"future_data_used_for_fit":False}}
