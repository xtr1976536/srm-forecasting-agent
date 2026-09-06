from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
import io
import zipfile
import urllib.request
import numpy as np
import pandas as pd

@dataclass
class MarketPanel:
    rv: pd.DataFrame
    returns: pd.DataFrame
    source: str
    is_proxy: bool
    iv: pd.DataFrame | None = None

def load_csv_panel(path):
    root=Path(path); rv_file=root/"merged_rv_data_filled.csv" if root.is_dir() else root
    if root.is_file() and root.suffix.lower()==".zip":
        with zipfile.ZipFile(root) as z:
            names=z.namelist()
            rv_name=next((n for n in names if n.endswith("merged_rv_data_filled.csv")),None)
            ret_name=next((n for n in names if n.endswith("daily_returns.csv")),None)
            iv_name=next((n for n in names if n.endswith("merged_iv_data_filled.csv")),None)
            if not rv_name: raise FileNotFoundError("ZIP is missing merged_rv_data_filled.csv")
            rv=pd.read_csv(io.BytesIO(z.read(rv_name)),parse_dates=["Date"]).set_index("Date").sort_index()
            returns=pd.read_csv(io.BytesIO(z.read(ret_name)),parse_dates=["Date"]).set_index("Date").sort_index() if ret_name else rv.pct_change()
            iv=pd.read_csv(io.BytesIO(z.read(iv_name)),parse_dates=["Date"]).set_index("Date").sort_index() if iv_name else None
    else:
        rv=pd.read_csv(rv_file,parse_dates=["Date"]).set_index("Date").sort_index()
        ret_file=root/"daily_returns.csv" if root.is_dir() else root.with_name("daily_returns.csv")
        returns=pd.read_csv(ret_file,parse_dates=["Date"]).set_index("Date").sort_index() if ret_file.exists() else rv.pct_change()
        iv_file=root/"merged_iv_data_filled.csv" if root.is_dir() else root.with_name("merged_iv_data_filled.csv")
        iv=pd.read_csv(iv_file,parse_dates=["Date"]).set_index("Date").sort_index() if iv_file.exists() else None
    returns=returns.replace([np.inf,-np.inf],np.nan).fillna(0.0)
    common=rv.index.intersection(returns.index).drop_duplicates().sort_values(); cols=sorted(set(rv.columns)&set(returns.columns))
    rv,returns=rv.loc[common,cols].astype(float),returns.loc[common,cols].astype(float)
    valid=np.isfinite(rv).all(axis=0)&(rv>0).all(axis=0)&np.isfinite(returns).all(axis=0); cols=[c for c,k in zip(cols,valid) if k]
    if not cols: raise ValueError("no complete positive RV series")
    if iv is not None:
        common=common.intersection(iv.index); cols=[c for c in cols if c in iv.columns]
        rv,returns,iv=rv.loc[common,cols],returns.loc[common,cols],iv.loc[common,cols].astype(float)
        iv_values=iv.to_numpy(float)
        valid_iv=np.isfinite(iv_values).all(axis=0)&(iv_values>0).all(axis=0)
        cols=[c for c,k in zip(cols,valid_iv) if k]
        rv,returns,iv=rv[cols],returns[cols],iv[cols]
    return MarketPanel(rv[cols],returns[cols],str(root),False,iv)

def fetch_yahoo_proxy(tickers,lookback_days=3000):
    import yfinance as yf
    end=datetime.utcnow(); start=end-timedelta(days=lookback_days)
    raw=yf.download(tickers,start=start.date(),end=end.date(),auto_adjust=True,progress=False,threads=True)
    if raw.empty: raise ValueError("Yahoo Finance returned no data")
    close=raw["Close"] if isinstance(raw.columns,pd.MultiIndex) else raw[["Close"]].rename(columns={"Close":tickers[0]})
    returns=close.dropna(how="all").ffill().pct_change().replace([np.inf,-np.inf],np.nan).dropna(how="all")
    rv=returns.pow(2).rolling(5).sum().mul(252/5).dropna(how="all"); common=rv.index.intersection(returns.index); cols=[c for c in rv.columns if c in returns.columns]
    return MarketPanel(rv.loc[common,cols],returns.loc[common,cols],"Yahoo Finance daily Close",True)

def fetch_stooq_proxy(tickers,lookback_days=1800):
    """Public CSV fallback. Stooq data are daily; results remain RV proxies."""
    frames=[]
    end=datetime.utcnow().date(); start=(datetime.utcnow()-timedelta(days=lookback_days)).date()
    for ticker in tickers:
        symbol=ticker.lower().replace('.','-')
        url=f"https://stooq.com/q/d/l/?s={symbol}.us&i=d&d1={start:%Y%m%d}&d2={end:%Y%m%d}"
        try:
            frame=pd.read_csv(url,parse_dates=["Date"]).set_index("Date")["Close"].rename(ticker); frames.append(frame)
        except Exception:
            continue
    if not frames: raise ValueError("Public fallback returned no valid ticker data")
    close=pd.concat(frames,axis=1).sort_index().ffill(); returns=close.pct_change().replace([np.inf,-np.inf],np.nan).dropna(how="all")
    rv=returns.pow(2).rolling(5).sum().mul(252/5).dropna(how="all"); common=rv.index.intersection(returns.index); cols=[c for c in rv.columns if c in returns.columns]
    return MarketPanel(rv.loc[common,cols],returns.loc[common,cols],"Stooq daily Close",True)
