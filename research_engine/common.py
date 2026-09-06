"""Shared data, target, window, and loss definitions for all five models.

Every runner imports this module.  This is deliberate: the Date--Ticker--
Horizon keys and future-RV target cannot be reimplemented independently by
the baseline and retrieval models.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import io
import json
import zipfile

import numpy as np
import pandas as pd


EPS = 1e-8
LAG = 22


@dataclass(frozen=True)
class Protocol:
    lag: int = LAG
    train_size: int = 600
    validation_size: int = 200
    test_size: int = 21
    step_size: int = 21
    horizons: tuple[int, ...] = (1, 5, 21)
    seed: int = 42


@dataclass(frozen=True)
class Window:
    horizon: int
    window_id: int
    train_t: np.ndarray
    validation_t: np.ndarray
    test_t: np.ndarray

    @property
    def fit_t(self) -> np.ndarray:
        """The estimation sample. Validation is never included here."""
        return self.train_t


@dataclass(frozen=True)
class Panel:
    rv: pd.DataFrame
    iv: pd.DataFrame
    returns: pd.DataFrame

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.rv.index

    @property
    def tickers(self) -> list[str]:
        return list(self.rv.columns)


def _read_csv_from_zip(archive: zipfile.ZipFile, suffixes: tuple[str, ...]) -> pd.DataFrame:
    name = next((n for n in archive.namelist() if any(n.endswith(s) for s in suffixes)), None)
    if name is None:
        raise FileNotFoundError(f"archive is missing one of {suffixes}")
    return pd.read_csv(io.BytesIO(archive.read(name)), parse_dates=["Date"]).set_index("Date").sort_index()


def _read_csv_from_directory(root: Path, names: tuple[str, ...]) -> pd.DataFrame:
    file = next((p for name in names for p in root.rglob(name)), None)
    if file is None:
        raise FileNotFoundError(f"{root} is missing one of {names}")
    return pd.read_csv(file, parse_dates=["Date"]).set_index("Date").sort_index()


def load_complete_panel(source: Path) -> tuple[Panel, dict]:
    """Load and strictly align RV, IV, and return panels.

    A ticker is retained only if all three data sources are finite on every
    common date, RV and IV are strictly positive, and dates are unique.  This
    makes the information universe identical for SRM and every benchmark,
    even though SRM itself only consumes RV.
    """
    source = Path(source)
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            rv = _read_csv_from_zip(archive, ("merged_rv_data_filled.csv",))
            iv = _read_csv_from_zip(archive, ("merged_iv_data_filled.csv",))
            ret = _read_csv_from_zip(archive, ("daily_returns.csv", "dow30_daily_returns_2021_2026.csv"))
    elif source.is_dir():
        rv = _read_csv_from_directory(source, ("merged_rv_data_filled.csv",))
        iv = _read_csv_from_directory(source, ("merged_iv_data_filled.csv",))
        ret = _read_csv_from_directory(source, ("daily_returns.csv", "dow30_daily_returns_2021_2026.csv"))
    else:
        raise FileNotFoundError(f"data source does not exist: {source}")

    for name, frame in (("RV", rv), ("IV", iv), ("returns", ret)):
        if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
            raise ValueError(f"{name} dates must be unique and increasing")

    dates = rv.index.intersection(iv.index).intersection(ret.index).sort_values()
    columns = sorted(set(rv.columns).intersection(iv.columns).intersection(ret.columns))
    if not len(dates) or not columns:
        raise ValueError("no common dates or tickers")
    rv, iv, ret = rv.loc[dates, columns], iv.loc[dates, columns], ret.loc[dates, columns]

    rv_values, iv_values, ret_values = rv.to_numpy(float), iv.to_numpy(float), ret.to_numpy(float)
    missing_counts = {
        "RV": int((~np.isfinite(rv_values)).sum()),
        "IV": int((~np.isfinite(iv_values)).sum()),
        "returns": int((~np.isfinite(ret_values)).sum()),
    }
    valid = (
        np.isfinite(rv_values).all(axis=0)
        & (rv_values > 0).all(axis=0)
        & np.isfinite(iv_values).all(axis=0)
        & (iv_values > 0).all(axis=0)
        & np.isfinite(ret_values).all(axis=0)
    )
    retained = [ticker for ticker, keep in zip(columns, valid) if keep]
    dropped = [ticker for ticker, keep in zip(columns, valid) if not keep]
    if not retained:
        raise ValueError("complete-case filtering retained no tickers")
    panel = Panel(rv.loc[:, retained], iv.loc[:, retained], ret.loc[:, retained])
    audit = {
        "source": str(source),
        "common_dates": int(len(panel.dates)),
        "source_date_counts": {"RV": int(len(rv.index)), "IV": int(len(iv.index)), "returns": int(len(ret.index))},
        "common_ticker_count_before_filter": int(len(columns)),
        "missing_or_nonfinite_counts_before_filter": missing_counts,
        "invalid_or_nonpositive_rv_count": int((~np.isfinite(rv_values) | (rv_values <= 0)).sum()),
        "invalid_or_nonpositive_iv_count": int((~np.isfinite(iv_values) | (iv_values <= 0)).sum()),
        "date_start": str(panel.dates.min().date()),
        "date_end": str(panel.dates.max().date()),
        "retained_tickers": retained,
        "dropped_tickers": dropped,
    }
    return panel, audit


def future_target(rv: np.ndarray, t: int, horizon: int) -> np.ndarray:
    """Target RV_t,...,RV_(t+h-1) for a forecast formed after t-1."""
    out = np.asarray(rv, dtype=float)[t : t + horizon]
    if out.shape[0] != horizon:
        raise ValueError(f"incomplete target at t={t}, h={horizon}")
    return out.mean(axis=0)


def make_windows(n_dates: int, protocol: Protocol, horizon: int) -> list[Window]:
    """Create one authoritative index-space window list.

    The panel index starts at `lag`, because every forecast must have the
    preceding 22 RV observations.  The two purge blocks contain exactly h
    valid forecast origins and are excluded from every estimation sample.
    """
    if horizon not in protocol.horizons:
        raise ValueError(f"unsupported horizon {horizon}")
    panel_orig = np.arange(protocol.lag, n_dates - horizon + 1, dtype=int)
    needed = protocol.train_size + horizon + protocol.validation_size + horizon + protocol.test_size
    windows: list[Window] = []
    for start in range(0, len(panel_orig) - needed + 1, protocol.step_size):
        tr = panel_orig[start : start + protocol.train_size]
        va_start = start + protocol.train_size + horizon
        va = panel_orig[va_start : va_start + protocol.validation_size]
        te_start = va_start + protocol.validation_size + horizon
        te = panel_orig[te_start : te_start + protocol.test_size]
        if len(tr) != protocol.train_size or len(va) != protocol.validation_size or len(te) != protocol.test_size:
            continue
        windows.append(Window(horizon, len(windows), tr, va, te))
    if not windows:
        raise ValueError(f"insufficient dates ({n_dates}) for horizon {horizon}")
    return windows


def windows_frame(windows: list[Window], dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for w in windows:
        rows.append({
            "Horizon": w.horizon,
            "Window_ID": w.window_id,
            "Train_Start": dates[w.train_t[0]], "Train_End": dates[w.train_t[-1]],
            "Validation_Start": dates[w.validation_t[0]], "Validation_End": dates[w.validation_t[-1]],
            "Test_Start": dates[w.test_t[0]], "Test_End": dates[w.test_t[-1]],
            "Purge": w.horizon,
        })
    return pd.DataFrame(rows)


def har_features(rv: np.ndarray, iv: np.ndarray, t: int, use_iv: bool) -> np.ndarray:
    """The common HARM feature map at forecast origin t (information through t-1)."""
    x = np.column_stack((
        rv[t - 1],
        rv[t - 5 : t - 1].mean(axis=0),
        rv[t - 22 : t - 5].mean(axis=0),
    ))
    if use_iv:
        x = np.column_stack((x, iv[t - 1]))
    return x


def qlike_np(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    p = np.maximum(np.asarray(prediction, float), EPS)
    ratio = np.asarray(y, float) / p
    return ratio - np.log(ratio) - 1.0


def metric_row(y: np.ndarray, prediction: np.ndarray) -> dict:
    error = np.asarray(y, float) - np.asarray(prediction, float)
    return {
        "MSE": float(np.mean(error * error)),
        "QLIKE": float(np.mean(qlike_np(y, prediction))),
        "MAE": float(np.mean(np.abs(error))),
        "N": int(np.size(y)),
    }


def write_json(path: Path, object_: dict) -> None:
    path.write_text(json.dumps(object_, indent=2, default=str), encoding="utf-8")


def protocol_dict(protocol: Protocol) -> dict:
    return asdict(protocol)
