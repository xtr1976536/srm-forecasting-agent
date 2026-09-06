import numpy as np
import pandas as pd

from engine import forecast
from intraday import _risk_curve
from storage import Store
from agent import SRMForecastingAgent


def panel(n=80):
    rng=np.random.default_rng(42)
    return pd.DataFrame(np.exp(rng.normal(-4,.2,(n,3))),index=pd.date_range("2025-01-01",periods=n),columns=["AAA","BBB","CCC"])


def test_forecast_is_positive_and_target_specific():
    result=forecast(panel(),["AAA","BBB","CCC"],horizon=5,k=10,scope="cross_asset")
    assert all(np.isfinite(v) and v>0 for v in result["predictions"].values())
    assert all(np.isclose(sum(t["weights"]),1.0) for t in result["traces"].values())


def test_same_asset_never_uses_other_assets():
    result=forecast(panel(),["AAA","BBB"],horizon=1,k=10,scope="same_asset")
    for target,trace in result["traces"].items():
        assert {n["ticker"] for n in trace["neighbors"]}=={target}


def test_intraday_curve_is_provisional_and_finite():
    frame=pd.DataFrame({"price":[100,101,100.5]},index=pd.date_range("2026-01-02 09:30",periods=3,freq="5min"))
    out=_risk_curve(frame,"test")
    assert out.attrs["provisional"] is True
    assert np.isfinite(out["annualized_vol"].dropna()).all()


def test_paper_account_constraints(tmp_path):
    store=Store(tmp_path/"paper.sqlite3"); account=store.create_account(1000)
    account=store.order(account["account_id"],"AAA","buy",2,100,.001)
    assert account["positions"]["AAA"]==2
    try: store.order(account["account_id"],"AAA","sell",3,100,.001)
    except ValueError: pass
    else: raise AssertionError("overselling must fail")

def test_agent_parses_user_parameters(tmp_path):
    task=SRMForecastingAgent(tmp_path).parse_task("forecast AAPL MSFT horizon=21 K=60 same asset criterion=MSE")
    assert task["horizon"]==21 and task["k"]==60 and task["cross_asset"] is False and task["criterion"]=="mse"
