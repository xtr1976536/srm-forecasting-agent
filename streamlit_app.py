from __future__ import annotations
import json
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from agent import SRMForecastingAgent
from intraday import yahoo_intraday, alpha_vantage_intraday

st.set_page_config(page_title="SRM Volatility Agent", page_icon="~", layout="wide")
agent = SRMForecastingAgent("srm_agent_runs")

st.markdown("""<style>
[data-testid="stAppViewContainer"]{background:#f5f7fa}.stApp{color:#172b3a}
[data-testid="stSidebar"]{background:#102a43;color:white}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p{color:#d9e7f2!important}
h1,h2,h3{letter-spacing:0!important}.block-container{padding-top:2rem;max-width:1500px}
[data-testid="stMetric"]{background:white;border:1px solid #d9e2ec;padding:14px;border-radius:6px}
</style>""",unsafe_allow_html=True)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_online_forecast(request_text):
    return agent.run(request_text)

@st.cache_data(ttl=240, show_spinner=False)
def cached_intraday(ticker, provider, key=""):
    return alpha_vantage_intraday(ticker,key) if provider=="Alpha Vantage" else yahoo_intraday(ticker)

st.title("SRM Volatility Forecasting Agent")
st.caption("Daily volatility forecasts from geometrically similar market histories")
st.warning("Research and paper-trading only. Not investment advice. Public market data produce a daily RV proxy, not high-frequency realized volatility.")
st.info("SRM forecasts volatility magnitude, not price direction. Use the output for risk sizing and simulation; a separate return model would be required for a bullish/bearish price forecast.")

with st.expander("How to use / 使用说明", expanded=True):
    st.markdown("""
    **1. Choose data and assets.** Use Yahoo for a quick public-data demonstration, or upload your own realized-volatility CSV/ZIP. Enter several tickers for cross-asset retrieval.

    **2. Choose a horizon.** `1`, `5`, and `21` mean the next trading day or the average volatility over the next 5 or 21 trading days. `K` is the number of similar historical paths used by SRM.

    **3. Run and interpret.** Lower predicted RV means a calmer expected volatility state; higher predicted RV means greater expected risk. Compare models, then inspect the historical analogs and audit record. The forecast updates meaningfully only after a new daily observation becomes available.

    **中文：** 先选择数据和股票，再选择预测期限与近邻数，最后点击 **Run daily forecast**。预测值表示未来期限内的平均波动率，而不是价格涨跌方向。
    """)

with st.sidebar:
    st.header("Daily forecast cycle")
    st.info("SRM is a daily model. Re-run after the market data source publishes a new daily close.")
    st.caption("No intraday auto-refresh: repeating the model before a new daily observation would not add information.")
    st.divider()
    st.header("Intraday monitor")
    intraday_refresh=st.toggle("Refresh while this page is open",value=True)
    intraday_minutes=st.select_slider("Interval (minutes)",options=[1,5,10,15],value=5)
    if intraday_refresh: st_autorefresh(interval=intraday_minutes*60*1000,key="intraday_refresh")

if "paper" not in st.session_state:
    st.session_state.paper = {"cash": 100000.0, "initial_cash": 100000.0, "positions": {}, "orders": [], "marks": {}, "equity_curve": []}

forecast_tab, intraday_tab, lab_tab, analog_tab, agent_tab, paper_tab, audit_tab = st.tabs(["Forecast Monitor", "Intraday Risk", "Model Lab", "Analog Explorer", "Agent", "Paper Trading", "Audit"])

with forecast_tab:
    left, right = st.columns([1, 2])
    with left:
        data_mode = st.selectbox("Data source", ["Yahoo daily RV proxy", "User RV dataset"])
        uploaded = st.file_uploader("Upload RV CSV (optional)", type=["csv","zip"]) if data_mode == "User RV dataset" else None
        tickers = st.text_input("Tickers", "AAPL,MSFT,NVDA")
        horizon = st.selectbox("Forecast horizon", [1, 5, 21], index=1)
        neighbors = st.number_input("Neighbors K", min_value=5, max_value=100, value=20, step=5)
        scope = st.selectbox("Retrieval scope", ["cross_asset", "same_asset"])
        criterion = st.selectbox("Estimation criterion", ["QLIKE", "MSE"])
        selected_models = st.multiselect("Models", ["srm", "historical_mean", "har", "har_iv", "gharm", "gharm_iv"], default=["srm", "historical_mean", "har"])
        st.caption("Cross-asset retrieval searches analog paths across all entered tickers. Same-asset retrieval searches only each target's own history.")
        run = st.button("Run daily forecast", type="primary", use_container_width=True)
    with right:
        if run:
            symbols = [x.strip().upper() for x in tickers.split(",") if x.strip()]
            try:
                if len(symbols) > 5:
                    raise ValueError("The free full-SRM service supports at most 5 tickers per run. Use an uploaded research job for larger universes.")
                csv_path = None
                if uploaded is not None:
                    suffix = ".zip" if uploaded.name.lower().endswith(".zip") else ".csv"
                    tmp = Path("/tmp/srm_uploaded" + suffix); tmp.write_bytes(uploaded.getvalue()); csv_path = str(tmp)
                    st.session_state.research_data_path=csv_path
                request_text = f"forecast {' '.join(symbols)} horizon={horizon} k={neighbors} {scope} criterion={criterion.lower()}"
                st.session_state.last_request = request_text
                with st.spinner("Building geometric memory, fitting channel weights, and retrieving analog paths..."):
                    result = agent.run(request_text, csv_path) if csv_path else cached_online_forecast(request_text)
                st.session_state.last_result = result
            except Exception as exc:
                st.error(str(exc))
        result = st.session_state.get("last_result")
        if result:
            source=result.get("data_audit",{}).get("source","Unknown")
            st.success(f"Data through {result['forecast_date']} | {result['horizon']}-day average forecast | {source}")
            st.caption(f"Last successful update: {result.get('data_timestamp','not recorded')} | Run ID: {result.get('run_id','')}")
            st.subheader("Forecast summary")
            cols=st.columns(min(4,len(result["predictions"])))
            for idx,(ticker,value) in enumerate(result["predictions"].items()):
                cols[idx%len(cols)].metric(ticker, f"{value:.6f}")
            if result.get("warnings"):
                for warning in result["warnings"]: st.warning(warning)
            st.subheader("Model comparison")
            rows=[]
            for model, values in result.get("model_predictions", {}).items():
                if model not in selected_models: continue
                if values is None:
                    for ticker in result["predictions"]: rows.append({"Model":model,"Ticker":ticker,"Forecast RV":"Unavailable: requires IV/returns research data"})
                else:
                    for ticker, value in values.items(): rows.append({"Model":model,"Ticker":ticker,"Forecast RV":value})
            st.dataframe(rows, use_container_width=True, hide_index=True)
            st.subheader("Historical volatility path and horizon-average forecast")
            fig = go.Figure()
            for ticker in result["predictions"]:
                hist = result.get("recent_rv", {}).get(ticker, [])
                if hist:
                    fig.add_trace(go.Scatter(x=[x["date"] for x in hist], y=[x["value"] for x in hist], mode="lines", name=f"{ticker} realized/proxy"))
                future = result.get("forecast_curve", {}).get(ticker, [])
                if future:
                    fig.add_trace(go.Scatter(x=[f"Forecast h={result['horizon']}"] , y=[future[0]["value"]], mode="markers", marker=dict(size=12), name=f"{ticker} forecast average"))
            fig.update_layout(height=380, margin=dict(l=10,r=10,t=20,b=10), xaxis_title="Date / forecast horizon", yaxis_title="RV or daily-data RV proxy", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("The point labelled Forecast is the predicted average RV over the selected horizon. SRM does not claim to predict a separate daily path inside that horizon.")
            st.subheader("Retrieved analog paths")
            if result.get("neighbors"):
                st.dataframe(result["neighbors"][:10], use_container_width=True, hide_index=True)
            elif result.get("diagnostics"):
                st.caption("Full five-channel SRM diagnostics for the latest test origin")
                st.dataframe(result["diagnostics"][:20], use_container_width=True, hide_index=True)
            st.download_button("Download run JSON", json.dumps(result, indent=2), file_name=f"{result['run_id']}.json", mime="application/json")

with intraday_tab:
    st.subheader("Intraday risk monitor")
    st.caption("This monitor observes the current session. It does not retrain the daily SRM and does not predict price direction.")
    c1,c2=st.columns([1,2])
    with c1:
        live_ticker=st.text_input("Intraday ticker","AAPL")
        provider=st.selectbox("Intraday provider",["Yahoo public","Alpha Vantage"])
        if provider=="Alpha Vantage":
            st.info("Configure ALPHA_VANTAGE_API_KEY in Streamlit Secrets. Keys are never entered into the public page.")
        load_live=st.button("Load intraday monitor",type="primary")
    with c2:
        if load_live or st.session_state.get("intraday_loaded"):
            st.session_state.intraday_loaded=True
            try:
                key=st.secrets.get("ALPHA_VANTAGE_API_KEY","") if provider=="Alpha Vantage" else ""
                if provider=="Alpha Vantage" and not key: raise ValueError("Alpha Vantage key is not configured; select Yahoo public or ask the site owner to add a secret.")
                live=cached_intraday(live_ticker.upper(),provider,key)
                latest=live.iloc[-1]; m1,m2,m3=st.columns(3); m1.metric("Last price",f"{latest['price']:.2f}"); m2.metric("Provisional daily RV",f"{latest['cumulative_rv']:.6f}"); m3.metric("Annualized intraday vol",f"{latest['annualized_vol']:.2%}")
                fig=go.Figure(); fig.add_trace(go.Scatter(x=live.index,y=live["price"],name="Price",yaxis="y1")); fig.add_trace(go.Scatter(x=live.index,y=live["annualized_vol"],name="Provisional volatility",yaxis="y2")); fig.update_layout(height=480,hovermode="x unified",yaxis=dict(title="Price"),yaxis2=dict(title="Annualized volatility",overlaying="y",side="right"),margin=dict(l=10,r=10,t=20,b=10)); st.plotly_chart(fig,use_container_width=True)
                st.caption(f"Source: {live.attrs.get('source')} | Fetched: {live.attrs.get('fetched_at')} | Current-session values are provisional.")
            except Exception as exc: st.error(str(exc))

with lab_tab:
    st.subheader("Model Lab")
    st.caption("Leakage-controlled rolling diagnostics. Full SRM is compute-intensive, so the free service evaluates at most five recent origins per run.")
    b1,b2=st.columns([1,2])
    with b1:
        bt_tickers=st.text_input("Backtest tickers","AAPL,MSFT",key="bt_tickers")
        bt_model=st.selectbox("Backtest model",["srm","har","gharm","har_iv","gharm_iv"])
        bt_h=st.selectbox("Backtest horizon",[1,5,21],index=1,key="bt_h")
        bt_k=st.number_input("Backtest K",5,100,20,5)
        bt_scope=st.selectbox("Backtest retrieval scope",["cross_asset","same_asset"])
        bt_criterion=st.selectbox("Backtest criterion",["qlike","mse"])
        bt_origins=st.slider("Recent forecast origins",1,5,3)
        bt_run=st.button("Run backtest",type="primary")
    with b2:
        if bt_run:
            request=f"forecast {' '.join(x.strip().upper() for x in bt_tickers.split(',') if x.strip())} horizon={bt_h} k={bt_k} {bt_scope} criterion={bt_criterion}"
            try:
                with st.spinner("Running leakage-controlled rolling evaluation..."):
                    bt=agent.backtest(request,st.session_state.get("research_data_path"),bt_model,bt_origins)
                st.session_state.backtest_result=bt
            except Exception as exc: st.error(str(exc))
        bt=st.session_state.get("backtest_result")
        if bt:
            m1,m2,m3=st.columns(3); m1.metric("MSE",f"{bt['summary']['mse']:.6g}"); m2.metric("QLIKE",f"{bt['summary']['qlike']:.6g}"); m3.metric("MAE",f"{bt['summary']['mae']:.6g}")
            btdf=pd.DataFrame(bt["predictions"]); fig=go.Figure(); fig.add_trace(go.Scatter(x=btdf["Date"],y=btdf["Target"],name="Realized")); fig.add_trace(go.Scatter(x=btdf["Date"],y=btdf["Prediction"],name="Forecast")); fig.update_layout(height=420,hovermode="x unified",margin=dict(l=10,r=10,t=20,b=10)); st.plotly_chart(fig,use_container_width=True)
            st.dataframe(btdf,use_container_width=True,hide_index=True); st.download_button("Download backtest CSV",btdf.to_csv(index=False),"backtest.csv","text/csv")

with analog_tab:
    st.subheader("Analog Explorer")
    result = st.session_state.get("last_result")
    if result and result.get("neighbors"):
        st.dataframe(result["neighbors"], use_container_width=True, hide_index=True)
        st.bar_chart(pd.DataFrame({"weight":result["weights"]}, index=[f"{n['ticker']}@{n['endpoint']}" for n in result["neighbors"]]))
    elif result:
        st.dataframe(result.get("diagnostics", []), use_container_width=True, hide_index=True)
    else: st.info("Run a forecast first.")

with agent_tab:
    st.subheader("Forecast agent")
    st.caption("Describe the analysis you need. The agent resolves parameters and calls auditable numerical tools; it does not invent forecast values.")
    prompt=st.text_area("Task", "Compare AAPL, MSFT and NVDA for the next 5 trading days using cross-asset SRM with K=20.", height=100)
    resolved=agent.parse_task(prompt)
    st.code(json.dumps(resolved,indent=2),language="json")
    if st.button("Run agent task",type="primary"):
        try:
            with st.spinner("Running the resolved forecasting workflow..."):
                agent_result=cached_online_forecast(prompt)
            st.session_state.last_result=agent_result
            st.success(f"Completed {agent_result['run_id']}")
            st.dataframe([{"Ticker":k,"Predicted RV":v} for k,v in agent_result["predictions"].items()],use_container_width=True,hide_index=True)
            st.json({"warnings":agent_result.get("warnings",[]),"audit":agent_result.get("data_audit",{})})
        except Exception as exc: st.error(str(exc))

with paper_tab:
    st.subheader("Volatility-aware paper portfolio")
    st.caption("Volatility does not predict direction. This workspace uses forecasts to scale risk and records only simulated orders.")
    cash = st.number_input("Reset cash", min_value=1000.0, value=float(st.session_state.paper["initial_cash"]), step=1000.0)
    if st.button("Reset account"):
        st.session_state.paper={"cash":cash,"initial_cash":cash,"positions":{},"orders":[],"marks":{},"equity_curve":[]}
        st.rerun()
    p=st.session_state.paper
    st.metric("Cash", f"${p['cash']:,.2f}")
    result=st.session_state.get("last_result")
    if result:
        st.subheader("Risk budget calculator")
        risk_ticker=st.selectbox("Forecast asset",list(result["predictions"]),key="risk_ticker")
        target_risk=st.slider("Target volatility budget",0.05,0.30,0.12,0.01)
        predicted=max(float(result["predictions"][risk_ticker]),1e-8)
        scale=min(target_risk/predicted,1.0)
        suggested=p["cash"]*scale
        c1,c2,c3=st.columns(3); c1.metric("Predicted RV",f"{predicted:.4f}"); c2.metric("Risk scale",f"{scale:.1%}"); c3.metric("Max simulated allocation",f"${suggested:,.0f}")
        st.caption("This is a volatility-targeting illustration, not a buy/sell recommendation. Directional exposure must be chosen separately.")
        if st.button("Apply target allocation (simulated)"):
            mark=float(st.session_state.get("last_prices",{}).get(risk_ticker,0))
            if mark<=0: st.warning("Enter an execution price below before applying the allocation.")
            else:
                target_value=suggested; current_value=p["positions"].get(risk_ticker,0)*mark; delta=max(target_value-current_value,0); shares=delta/mark
                if shares>0 and delta*(1+0.0005)<=p["cash"]:
                    fee_cost=delta*0.0005; p["cash"]-=delta+fee_cost; p["positions"][risk_ticker]=p["positions"].get(risk_ticker,0)+shares; p["marks"][risk_ticker]=mark; p["orders"].append({"ticker":risk_ticker,"side":"buy","quantity":shares,"price":mark,"fee":fee_cost,"reason":"volatility_target"}); st.success(f"Added {shares:.4f} shares in simulation")
                else: st.warning("No allocation added: target is already met or cash is insufficient.")
    oticker=st.text_input("Order ticker", "AAPL")
    side=st.selectbox("Side", ["buy","sell"]); qty=st.number_input("Quantity", min_value=0.0001, value=1.0); price=st.number_input("Execution price", min_value=0.0001, value=100.0); fee=st.number_input("Transaction cost", min_value=0.0, max_value=0.1, value=0.0005, format="%.4f")
    if st.button("Place simulated order"):
        gross=qty*price; cost=gross*fee; held=p["positions"].get(oticker.upper(),0)
        if side=="buy" and gross+cost>p["cash"]: st.error("Insufficient cash")
        elif side=="sell" and qty>held: st.error("Insufficient position")
        else:
            p["cash"] += (-gross-cost if side=="buy" else gross-cost); p["positions"][oticker.upper()]=held+(qty if side=="buy" else -qty); p["marks"][oticker.upper()]=price; p["orders"].append({"ticker":oticker.upper(),"side":side,"quantity":qty,"price":price,"fee":cost,"reason":"manual"}); st.success("Simulated order recorded")
    p["marks"][oticker.upper()]=price
    st.session_state.last_prices=p["marks"]
    market_value=sum(q*p["marks"].get(t,0) for t,q in p["positions"].items()); nav=p["cash"]+market_value; p["equity_curve"].append({"step":len(p["equity_curve"])+1,"nav":nav}); curve=pd.DataFrame(p["equity_curve"]); peak=curve["nav"].cummax(); drawdown=(curve["nav"]/peak-1).min() if len(curve) else 0
    c1,c2,c3=st.columns(3); c1.metric("Net asset value",f"${nav:,.2f}"); c2.metric("Return",f"{nav/p['initial_cash']-1:.2%}"); c3.metric("Max drawdown",f"{drawdown:.2%}")
    if len(curve)>1: st.plotly_chart(go.Figure(go.Scatter(x=curve["step"],y=curve["nav"],mode="lines+markers",name="Paper NAV")).update_layout(height=280,margin=dict(l=10,r=10,t=20,b=10)),use_container_width=True)
    st.subheader("Positions"); st.json(p["positions"]); st.subheader("Orders"); st.dataframe(p["orders"], use_container_width=True, hide_index=True)

with audit_tab:
    st.subheader("Information-set audit")
    if st.session_state.get("last_result"):
        st.json(st.session_state.last_result.get("data_audit", {}))
        st.info("The online demo uses daily Yahoo data as an RV proxy. Paper-grade experiments should use the research RV adapter and the frozen paper protocol.")
    else:
        st.info("Run a forecast first to view its audit record.")
