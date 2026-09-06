from __future__ import annotations

import json
from pathlib import Path
from fastapi import FastAPI, Query
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
<main><div class="grid"><section class="panel"><h2>Forecast</h2><label>Tickers</label><input id="tickers" value="AAPL,MSFT,NVDA"/><label>Horizon</label><select id="h"><option value="1">1 trading day</option><option value="5" selected>5 trading days</option><option value="21">21 trading days</option></select><label>Neighbors (K)</label><input id="k" type="number" value="20" min="5" max="100"/><label>Retrieval scope</label><select id="scope"><option value="cross_asset" selected>Cross-asset</option><option value="same_asset">Same-asset</option></select><button onclick="runForecast()">Run Forecast</button><p class="muted">Yahoo daily data is labeled as an RV proxy. Use licensed intraday data for publication-grade RV.</p></section>
<section class="panel"><div id="status" class="muted">Choose assets and run a forecast.</div><div id="results" class="hidden"><h2>Forecast Results</h2><div id="cards"></div><canvas id="chart" height="130"></canvas><h3>Retrieved analog paths</h3><div id="neighbors"></div><h3>Paper-trading note</h3><p class="muted">This dashboard supports simulated research only. It does not place real orders or provide investment advice.</p></div></section></div></main>
<script>
let chart;
async function runForecast(){const status=document.getElementById('status');const results=document.getElementById('results');status.textContent='Fetching market data and retrieving paths...';results.classList.add('hidden');const q=new URLSearchParams({tickers:document.getElementById('tickers').value,h:document.getElementById('h').value,k:document.getElementById('k').value,scope:document.getElementById('scope').value});try{const r=await fetch('/api/forecast?'+q);const d=await r.json();if(!r.ok)throw new Error(d.detail||'forecast failed');document.getElementById('cards').innerHTML=Object.entries(d.predictions).map(([t,v])=>`<span class="metric"><b>${t}</b><br>${Number(v).toPrecision(6)}</span>`).join('');document.getElementById('neighbors').innerHTML=d.neighbors.slice(0,10).map((n,i)=>`<div class="neighbor"><b>${i+1}. ${n.ticker}</b> endpoint ${n.endpoint}, distance ${Number(n.distance).toFixed(4)}, weight ${Number(d.weights[i]).toFixed(3)}, future change ${Number(n.relative_change).toFixed(4)}</div>`).join('');if(chart)chart.destroy();chart=new Chart(document.getElementById('chart'),{type:'bar',data:{labels:Object.keys(d.predictions),datasets:[{label:'Forecast RV',data:Object.values(d.predictions),backgroundColor:'#147d92'}]},options:{responsive:true,plugins:{legend:{display:false}}}});status.textContent=`Origin ${d.forecast_date} | ${d.horizon}-day forecast | ${d.candidate_count} eligible candidates`;results.classList.remove('hidden')}catch(e){status.innerHTML=`<div class="error">${e.message}</div>`}}
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
    result = agent.run(request)
    return result
