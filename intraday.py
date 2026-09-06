from __future__ import annotations
from datetime import datetime, timezone
import numpy as np
import pandas as pd

def yahoo_intraday(ticker: str, interval="5m", period="5d") -> pd.DataFrame:
    import yfinance as yf
    raw=yf.download(ticker,period=period,interval=interval,auto_adjust=True,progress=False,threads=False)
    if raw.empty: raise ValueError("No intraday bars returned")
    close=raw["Close"]
    if isinstance(close,pd.DataFrame): close=close.iloc[:,0]
    frame=pd.DataFrame({"price":close.astype(float)}).dropna()
    return _risk_curve(frame,"Yahoo Finance intraday (best-effort public feed)")

def alpha_vantage_intraday(ticker: str, api_key: str, interval="5min") -> pd.DataFrame:
    import requests
    response=requests.get("https://www.alphavantage.co/query",params={"function":"TIME_SERIES_INTRADAY","symbol":ticker,"interval":interval,"outputsize":"compact","apikey":api_key},timeout=20)
    response.raise_for_status(); payload=response.json(); key=next((k for k in payload if k.startswith("Time Series")),None)
    if not key: raise ValueError(payload.get("Note") or payload.get("Information") or payload.get("Error Message") or "No intraday data")
    close={pd.Timestamp(ts):float(row["4. close"]) for ts,row in payload[key].items()}
    return _risk_curve(pd.DataFrame({"price":pd.Series(close)}).sort_index(),"Alpha Vantage intraday")

def _risk_curve(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    out=frame.copy(); out["log_return"]=np.log(out["price"]).diff(); session=pd.Series(out.index.date,index=out.index)
    out["cumulative_rv"]=out["log_return"].pow(2).groupby(session).cumsum()
    out["annualized_vol"]=np.sqrt(252.0*out["cumulative_rv"])
    out.attrs.update({"source":source,"fetched_at":datetime.now(timezone.utc).isoformat(),"provisional":True})
    return out.dropna(subset=["price"])
