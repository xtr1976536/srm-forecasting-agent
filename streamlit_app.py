from __future__ import annotations
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
try:
    from srm_agent.agent import SRMForecastingAgent
except ModuleNotFoundError:
    from agent import SRMForecastingAgent

st.set_page_config(page_title="SRM Volatility Agent", page_icon="~", layout="wide")
agent = SRMForecastingAgent("srm_agent_runs")

st.title("SRM Volatility Forecasting Agent")
st.caption("Geometric path retrieval, model comparison and paper-trading research")
st.warning("Research and paper-trading purposes only. Not investment advice. Online Yahoo data is a daily-data RV proxy, not high-frequency realized volatility.")

with st.sidebar:
    st.header("Live monitor")
    auto_refresh = st.checkbox("Continuous refresh", value=True)
    refresh_seconds = st.slider("Refresh interval (minutes)", 1, 15, 5)
    st.caption("Refresh reruns the forecast with the latest available market data.")
if auto_refresh:
    st_autorefresh(interval=refresh_seconds * 60 * 1000, key="market_refresh")

if "paper" not in st.session_state:
    st.session_state.paper = {"cash": 100000.0, "initial_cash": 100000.0, "positions": {}, "orders": []}

forecast_tab, lab_tab, analog_tab, paper_tab, audit_tab = st.tabs(["Forecast Monitor", "Model Lab", "Analog Explorer", "Paper Trading", "Audit"])

with forecast_tab:
    left, right = st.columns([1, 2])
    with left:
        data_mode = st.selectbox("Data source", ["Yahoo daily RV proxy", "User RV dataset"])
        uploaded = st.file_uploader("Upload RV CSV (optional)", type=["csv","zip"]) if data_mode == "User RV dataset" else None
        tickers = st.text_input("Tickers", "AAPL,MSFT,NVDA")
        horizon = st.selectbox("Forecast horizon", [1, 5, 21], index=1)
        neighbors = st.number_input("Neighbors K", min_value=5, max_value=100, value=20, step=5)
        scope = st.selectbox("Retrieval scope", ["cross_asset", "same_asset"])
        run = st.button("Run forecast", type="primary", use_container_width=True)
    with right:
        if run:
            symbols = [x.strip().upper() for x in tickers.split(",") if x.strip()]
            try:
                csv_path = None
                if uploaded is not None:
                    suffix = ".zip" if uploaded.name.lower().endswith(".zip") else ".csv"
                    tmp = Path("/tmp/srm_uploaded" + suffix); tmp.write_bytes(uploaded.getvalue()); csv_path = str(tmp)
                result = agent.run(f"forecast {' '.join(symbols)} horizon={horizon} k={neighbors} {scope}", csv_path)
                st.session_state.last_result = result
            except Exception as exc:
                st.error(str(exc))
        result = st.session_state.get("last_result")
        if result:
            st.success(f"Origin: {result['forecast_date']} | {result['horizon']}-day horizon | Run: {result['run_id']}")
            st.subheader("SRM forecasts")
            st.dataframe([{ "Ticker": k, "SRM": v } for k, v in result["predictions"].items()], use_container_width=True, hide_index=True)
            st.subheader("Model comparison")
            rows=[]
            for model, values in result.get("model_predictions", {}).items():
                for ticker, value in values.items(): rows.append({"Model":model,"Ticker":ticker,"Forecast RV":value})
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.subheader("Forecast curve")
            history = st.session_state.get("history", [])
            history.append({"timestamp":result["forecast_date"], **result["predictions"]})
            st.session_state.history = history[-60:]
            curve_df = pd.DataFrame(st.session_state.history)
            fig = go.Figure()
            for ticker in result["predictions"]:
                fig.add_trace(go.Scatter(x=curve_df["timestamp"], y=curve_df[ticker], mode="lines+markers", name=ticker))
            fig.update_layout(height=340, margin=dict(l=10,r=10,t=20,b=10), xaxis_title="Forecast origin", yaxis_title="Predicted RV", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Retrieved analog paths")
            if result.get("neighbors"):
                st.dataframe(result["neighbors"][:10], use_container_width=True, hide_index=True)
            elif result.get("diagnostics"):
                st.caption("Full five-channel SRM diagnostics for the latest test origin")
                st.dataframe(result["diagnostics"][:20], use_container_width=True, hide_index=True)
            st.download_button("Download run JSON", json.dumps(result, indent=2), file_name=f"{result['run_id']}.json", mime="application/json")

with lab_tab:
    st.subheader("Model Lab")
    st.info("Use Forecast Monitor to run the latest online forecast. Research-data rolling backtests remain available through the frozen experiment runner.")
    result = st.session_state.get("last_result")
    if result:
        st.json({"run_id": result["run_id"], "horizon": result["horizon"], "data_audit": result["data_audit"], "models": list(result.get("model_predictions", {}))})
    else: st.info("Run a forecast first.")

with analog_tab:
    st.subheader("Analog Explorer")
    result = st.session_state.get("last_result")
    if result and result.get("neighbors"):
        st.dataframe(result["neighbors"], use_container_width=True, hide_index=True)
        st.bar_chart(pd.DataFrame({"weight":result["weights"]}, index=[f"{n['ticker']}@{n['endpoint']}" for n in result["neighbors"]]))
    elif result:
        st.dataframe(result.get("diagnostics", []), use_container_width=True, hide_index=True)
    else: st.info("Run a forecast first.")

with paper_tab:
    st.subheader("Virtual account")
    cash = st.number_input("Reset cash", min_value=1000.0, value=float(st.session_state.paper["initial_cash"]), step=1000.0)
    if st.button("Reset account"):
        st.session_state.paper={"cash":cash,"initial_cash":cash,"positions":{},"orders":[]}
        st.rerun()
    p=st.session_state.paper
    st.metric("Cash", f"${p['cash']:,.2f}")
    oticker=st.text_input("Order ticker", "AAPL")
    side=st.selectbox("Side", ["buy","sell"]); qty=st.number_input("Quantity", min_value=0.0001, value=1.0); price=st.number_input("Execution price", min_value=0.0001, value=100.0); fee=st.number_input("Transaction cost", min_value=0.0, max_value=0.1, value=0.0005, format="%.4f")
    if st.button("Place simulated order"):
        gross=qty*price; cost=gross*fee; held=p["positions"].get(oticker.upper(),0)
        if side=="buy" and gross+cost>p["cash"]: st.error("Insufficient cash")
        elif side=="sell" and qty>held: st.error("Insufficient position")
        else:
            p["cash"] += (-gross-cost if side=="buy" else gross-cost); p["positions"][oticker.upper()]=held+(qty if side=="buy" else -qty); p["orders"].append({"ticker":oticker.upper(),"side":side,"quantity":qty,"price":price,"fee":cost}); st.success("Simulated order recorded")
    st.subheader("Positions"); st.json(p["positions"]); st.subheader("Orders"); st.dataframe(p["orders"], use_container_width=True, hide_index=True)

with audit_tab:
    st.subheader("Information-set audit")
    if st.session_state.get("last_result"):
        st.json(st.session_state.last_result.get("data_audit", {}))
        st.info("The online demo uses daily Yahoo data as an RV proxy. Paper-grade experiments should use the research RV adapter and the frozen paper protocol.")
    else:
        st.info("Run a forecast first to view its audit record.")
