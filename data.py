from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
import io
import zipfile
import numpy as np
import pandas as pd

@dataclass
class MarketPanel:
    rv: pd.DataFrame
    returns: pd.DataFrame
    source: str
    is_proxy: bool

def load_csv_panel(path):
    root=Path(path); rv_file=root/"merged_rv_data_filled.csv" if root.is_dir() else root
    if root.is_file() and root.suffix.lower()==".zip":
        with zipfile.ZipFile(root) as z:
            names=z.namelist()
            rv_name=next((n for n in names if n.endswith("merged_rv_data_filled.csv")),None)
            ret_name=next((n for n in names if n.endswith("daily_returns.csv")),None)
            if not rv_name: raise FileNotFoundError("ZIP is missing merged_rv_data_filled.csv")
            rv=pd.read_csv(io.BytesIO(z.read(rv_name)),parse_dates=["Date"]).set_index("Date").sort_index()
            returns=pd.read_csv(io.BytesIO(z.read(ret_name)),parse_dates=["Date"]).set_index("Date").sort_index() if ret_name else rv.pct_change()
    else:
        rv=pd.read_csv(rv_file,parse_dates=["Date"]).set_index("Date").sort_index()
        ret_file=root/"daily_returns.csv" if root.is_dir() else root.with_name("daily_returns.csv")
        returns=pd.read_csv(ret_file,parse_dates=["Date"]).set_index("Date").sort_index() if ret_file.exists() else rv.pct_change()
    common=rv.index.intersection(returns.index).drop_duplicates().sort_values(); cols=sorted(set(rv.columns)&set(returns.columns))
    rv,returns=rv.loc[common,cols].astype(float),returns.loc[common,cols].astype(float)
    valid=np.isfinite(rv).all(axis=0)&(rv>0).all(axis=0)&np.isfinite(returns).all(axis=0); cols=[c for c,k in zip(cols,valid) if k]
    if not cols: raise ValueError("no complete positive RV series")
    return MarketPanel(rv[cols],returns[cols],str(root),False)

def fetch_yahoo_proxy(tickers,lookback_days=3000):
    import yfinance as yf
    end=datetime.utcnow(); start=end-timedelta(days=lookback_days)
    raw=yf.download(tickers,start=start.date(),end=end.date(),auto_adjust=True,progress=False,threads=True)
    if raw.empty: raise ValueError("Yahoo Finance returned no data")
    close=raw["Close"] if isinstance(raw.columns,pd.MultiIndex) else raw[["Close"]].rename(columns={"Close":tickers[0]})
    returns=close.dropna(how="all").ffill().pct_change().replace([np.inf,-np.inf],np.nan).dropna(how="all")
    rv=returns.pow(2).rolling(5).sum().mul(252/5).dropna(how="all"); common=rv.index.intersection(returns.index); cols=[c for c in rv.columns if c in returns.columns]
    return MarketPanel(rv.loc[common,cols],returns.loc[common,cols],"Yahoo Finance daily Close",True)
