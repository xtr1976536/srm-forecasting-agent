import numpy as np
import pandas as pd

from engine import forecast
from intraday import _risk_curve
from storage import Store
from agent import SRMForecastingAgent
from full_baselines import run_full_baselines
from backtest import run_backtest
from data import MarketPanel
from direction import return_scenario
from snapshot import anonymized_snapshot


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

def test_iv_baselines_use_iv_features():
    rng=np.random.default_rng(8); n=850; columns=["AAA","BBB","CCC"]; index=pd.date_range("2020-01-01",periods=n)
    rv=pd.DataFrame(np.exp(rng.normal(-4,.2,(n,3))),index=index,columns=columns)
    iv=pd.DataFrame(np.exp(rng.normal(-3,.5,(n,3))),index=index,columns=columns)
    returns=pd.DataFrame(rng.normal(0,.02,(n,3)),index=index,columns=columns)
    result=run_full_baselines(rv,iv,returns,columns,1,"mse")["predictions"]
    assert any(not np.isclose(result["har"][ticker],result["har_iv"][ticker]) for ticker in columns)

def test_backtest_uses_completed_future_only():
    rng=np.random.default_rng(10); n=850; columns=["AAA","BBB"]; index=pd.date_range("2020-01-01",periods=n)
    rv=pd.DataFrame(np.exp(rng.normal(-4,.2,(n,2))),index=index,columns=columns)
    returns=pd.DataFrame(rng.normal(0,.02,(n,2)),index=index,columns=columns)
    result=run_backtest(MarketPanel(rv,returns,"test",True),columns,1,"har",origins=2)
    assert result["audit"]["future_data_used_for_fit"] is False
    assert len(result["predictions"])==4

def test_return_scenario_and_public_snapshot_are_safe():
    scenario=return_scenario(np.array([0.01,-0.02,0.005]*10),100,5)
    assert scenario["lower_price"] < scenario["expected_price"] < scenario["upper_price"]
    result={"run_id":"run_x","forecast_date":"2026-01-01","horizon":5,"predictions":{"AAA":0.1},"task":{"criterion":"qlike"},"data_audit":{"is_daily_rv_proxy":True},"warnings":[]}
    snap=anonymized_snapshot(result)
    assert "cash" not in snap and "orders" not in snap and snap["audit_status"]=="passed"

def test_missing_returns_are_not_silently_filled(tmp_path):
    idx=pd.date_range("2025-01-01",periods=30); cols=["AAA"]
    rv=pd.DataFrame(np.ones((30,1))*0.01,index=idx,columns=cols); ret=pd.DataFrame(np.ones((30,1))*0.01,index=idx,columns=cols); ret.iloc[5,0]=np.nan
    rv.index.name=ret.index.name="Date"; rv.to_csv(tmp_path/"merged_rv_data_filled.csv"); ret.to_csv(tmp_path/"daily_returns.csv")
    try:
        from data import load_csv_panel
        load_csv_panel(tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete returns must be rejected")
