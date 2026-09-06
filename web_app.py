from __future__ import annotations

import json
from pathlib import Path
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from .agent import SRMForecastingAgent

app = FastAPI(title="SRM Volatility Forecasting Agent", version="0.1.0")
agent = SRMForecastingAgent("srm_agent_runs")

HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>SRM Forecasting Agent</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
body{font-family:Inter,system-ui,sans-serif;background:#f4f6f8;color:#17202a;margin:0}header{background:#102a43;color:#fff;padding:22px 8%;}main{max-width:1180px;margin:28px auto;padding:0 20px}.grid{display:grid;grid-template-columns:300px 1fr;gap:20px}.panel{background:#fff;border:1px solid #d9e2ec;border-radius:10px;padding:20px;box-shadow:0 2px 8px #102a4312}label{display:block;font-size:13px;font-weight:600;margin:13px 0 5px}input,select,button{width:100%;box-sizing:border-box;padding:10px;border:1px solid #bcccdc;border-radius:6px;font:inherit}button{background:#147d92;color:#fff;border:0;margin-top:18px;cursor:pointer;font-weight:700}button:hover{background:#0f6677}.metric{display:inline-block;background:#e8f1f5;border-radius:6px;padding:12px;margin:5px}.muted{color:#627d98;font-size:13px}.error{color:#b42318;background:#fff1f0;padding:12px;border-radius:6px}.neighbor{font-size:13px;border-bottom:1px solid #e6edf3;padding:8px 0}.hidden{display:none}@media(max-width:800px){.grid{grid-template-columns:1fr}}
</style></head><body><header><h1>SRM Volatility Forecasting Agent</h1><div>Geometric path retrieval, model comparison and paper-trading research</div></header>
<main><div class="grid"><section class="panel"><h2>Forecast</h2><label>Tickers</label><input id="tickers" value="AAPL,MSFT,NVDA"/><label>Horizon</label><select id="h"><option value="1">1 trading day</option><option value="5" selected>5 trading days</option><option value="21">21 trading days</option></select><label>Neighbors (K)</label><input id="k" type="number" value="20" min="5" max="100"/><label>Retrieval scope</label><select id="scope"><option value="cross_asset" selected>Cross-asset</option><option value="same_asset">Same-asset</option></select><button onclick="runForecast()">Run Forecast</button><hr><h2>Paper Account</h2><label>Initial cash</label><input id="cash" type="number" value="100000"/><button onclick="createAccount()">Create virtual account</button><div id="account" class="muted"></div><label>Order ticker</label><input id="oticker" value="AAPL"/><label>Side / quantity / price</label><select id="side"><option value="buy">Buy</option><option value="sell">Sell</option></select><input id="qty" type="number" value="1" min="0.0001" step="0.0001"/><input id="price" type="number" value="100" min="0.0001" step="0.0001"/><button onclick="placeOrder()">Place simulated order</button><p class="muted">Research and paper-trading purposes only. Not investment advice.</p></section>
<section class="panel"><div id="status" class="muted">Choose assets and run a forecast.</div><div id="results" class="hidden"><h2>Forecast Results</h2><div id="cards"></div><canvas id="chart" height="130"></canvas><h3>Model comparison</h3><div id="models"></div><h3>Retrieved analog paths</h3><div id="neighbors"></div><h3>Paper-trading note</h3><p class="muted">This dashboard supports simulated research only. It does not place real orders or provide investment advice.</p></div></section></div></main>
<script>
let chart;
let accountId;
async function createAccount(){const cash=document.getElementById('cash').value;const r=await fetch('/api/paper/account?cash='+cash,{method:'POST'});const d=await r.json();accountId=d.account_id;document.getElementById('account').textContent=`Account ${accountId} | cash ${Number(d.cash).toFixed(2)}`;}
async function placeOrder(){if(!accountId){alert('Create a virtual account first');return}const q=new URLSearchParams({account_id:accountId,ticker:document.getElementById('oticker').value,side:document.getElementById('side').value,quantity:document.getElementById('qty').value,price:document.getElementById('price').value});const r=await fetch('/api/paper/order?'+q,{method:'POST'});const d=await r.json();if(!r.ok){alert(d.detail||'order failed');return}document.getElementById('account').textContent=`Account ${d.account_id} | cash ${Number(d.cash).toFixed(2)} | positions ${JSON.stringify(d.positions)}`;}
async function runForecast(){const status=document.getElementById('status');const results=document.getElementById('results');status.textContent='Fetching market data and retrieving paths...';results.classList.add('hidden');const q=new URLSearchParams({tickers:document.getElementById('tickers').value,h:document.getElementById('h').value,k:document.getElementById('k').value,scope:document.getElementById('scope').value});try{const r=await fetch('/api/forecast?'+q);const d=await r.json();if(!r.ok)throw new Error(d.detail||'forecast failed');document.getElementById('cards').innerHTML=Object.entries(d.predictions).map(([t,v])=>`<span class="metric"><b>${t}</b><br>${Number(v).toPrecision(6)}</span>`).join('');document.getElementById('models').innerHTML=Object.entries(d.model_predictions).map(([m,p])=>`<div class="neighbor"><b>${m}</b>: `+Object.entries(p).map(([t,v])=>`${t} ${Number(v).toPrecision(5)}`).join(' | ')+`</div>`).join('');document.getElementById('neighbors').innerHTML=d.neighbors.slice(0,10).map((n,i)=>`<div class="neighbor"><b>${i+1}. ${n.ticker}</b> endpoint ${n.endpoint}, distance ${Number(n.distance).toFixed(4)}, weight ${Number(d.weights[i]).toFixed(3)}, future change ${Number(n.relative_change).toFixed(4)}</div>`).join('');if(chart)chart.destroy();chart=new Chart(document.getElementById('chart'),{type:'bar',data:{labels:Object.keys(d.predictions),datasets:[{label:'SRM Forecast RV',data:Object.values(d.predictions),backgroundColor:'#147d92'}]},options:{responsive:true,plugins:{legend:{display:false}}}});status.textContent=`Origin ${d.forecast_date} | ${d.horizon}-day forecast | ${d.candidate_count} eligible candidates | run ${d.run_id}`;results.classList.remove('hidden')}catch(e){status.innerHTML=`<div class="error">${e.message}</div>`}}
</script></body></html>"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML

@app.get("/api/forecast")
def api_forecast(tickers: str = Query(...), h: int = Query(1, ge=1, le=21), k: int = Query(20, ge=5, le=100), scope: str = Query("cross_asset")):
    symbols = [x.strip().upper() for x in tickers.split(",") if x.strip()]
    if not symbols:
        return {"detail": "Provide at least one ticker"}
    request = f"forecast {' '.join(symbols)} horizon={h} k={k} {scope}"
    try: result = agent.run(request)
    except (ValueError, RuntimeError) as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result

@app.get("/health")
def health(): return {"status":"ok","service":"srm-forecasting-agent"}

@app.get("/api/assets")
def assets():
    return {"assets":["AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AMD","JPM","SPY","QQQ"]}

@app.post("/api/agent/task")
def agent_task(request: dict):
    text=request.get("request") or request.get("prompt")
    if not text: raise HTTPException(status_code=400, detail="request or prompt is required")
    try: return agent.run(text, request.get("csv_path"))
    except (ValueError, RuntimeError) as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/api/backtest")
def backtest(request: dict):
    if not request.get("tickers"): raise HTTPException(status_code=400, detail="tickers are required")
    tickers=" ".join(str(x).upper() for x in request["tickers"])
    prompt=f"forecast {tickers} horizon={request.get('horizon',1)} k={request.get('neighbors',20)} {request.get('retrieval_scope','cross_asset')} criterion={request.get('criterion','qlike')}"
    try: return agent.backtest(prompt,request.get("csv_path"),request.get("model","srm"),min(int(request.get("origins",3)),5))
    except Exception as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/api/run/{run_id}")
def get_run(run_id: str):
    result=agent.store.run(run_id)
    if result is None: raise HTTPException(status_code=404, detail="run not found")
    return result

@app.post("/api/paper/account")
def create_account(cash: float = Query(100000.0, gt=0)): return agent.store.create_account(cash)

@app.get("/api/paper/account/{account_id}")
def get_account(account_id: str):
    result=agent.store.account(account_id)
    if result is None: raise HTTPException(status_code=404, detail="account not found")
    return result

@app.get("/api/paper/performance/{account_id}")
def paper_performance(account_id: str):
    result=agent.store.account(account_id)
    if result is None: raise HTTPException(status_code=404, detail="account not found")
    invested=result["initial_cash"]-result["cash"]
    return {"account_id":account_id,"cash":result["cash"],"initial_cash":result["initial_cash"],"invested_cash":invested,"orders":len(result["orders"]),"positions":result["positions"]}

@app.post("/api/paper/order")
def place_order(account_id: str, ticker: str, side: str, quantity: float, price: float, transaction_cost: float = 0.0005):
    try: return agent.store.order(account_id, ticker.upper(), side, quantity, price, transaction_cost)
    except ValueError as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc
