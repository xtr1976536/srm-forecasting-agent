import json,re
from pathlib import Path
from .data import fetch_yahoo_proxy,load_csv_panel
from .engine import forecast

class SRMForecastingAgent:
    def __init__(self,output_dir="srm_agent_runs"):
        self.output_dir=Path(output_dir); self.output_dir.mkdir(parents=True,exist_ok=True)
    def parse_task(self,request):
        text=request.lower(); hm=re.search(r"(?:horizon|h|未来)\s*[=:]?\s*(1|5|21)",text); km=re.search(r"(?:k|neighbor|近邻)\s*[=:]?\s*(\d+)",text); tickers=sorted(set(re.findall(r"\b[A-Z]{1,5}(?:\.[A-Z])?\b",request)))
        return {"horizon":int(hm.group(1)) if hm else 1,"k":int(km.group(1)) if km else 20,"tickers":tickers,"cross_asset":not any(x in text for x in ("same asset","同资产"))}
    def _audit(self,panel,horizon,k):
        if len(panel.rv)<23+horizon: raise ValueError("at least 23+h observations are required")
        if panel.rv.index.has_duplicates or not panel.rv.index.is_monotonic_increasing: raise ValueError("dates must be unique and increasing")
        if not panel.rv.columns.equals(panel.returns.columns): raise ValueError("RV and return tickers are not aligned")
        return {"date_start":str(panel.rv.index.min().date()),"date_end":str(panel.rv.index.max().date()),"n_dates":len(panel.rv),"n_tickers":len(panel.rv.columns),"horizon":horizon,"k":k,"is_daily_rv_proxy":panel.is_proxy,"source":panel.source}
    def run(self,request,csv_path=None):
        task=self.parse_task(request); panel=load_csv_panel(csv_path) if csv_path else fetch_yahoo_proxy(task["tickers"])
        if task["tickers"]:
            selected=[t for t in task["tickers"] if t in panel.rv.columns]
            if selected: panel=type(panel)(panel.rv[selected],panel.returns[selected],panel.source,panel.is_proxy)
        audit=self._audit(panel,task["horizon"],task["k"]); result=forecast(panel.rv,list(panel.rv.columns),task["horizon"],task["k"]); result.update({"task":task,"data_audit":audit,"model":"SRM-v1-curve-retrieval-demo"})
        out=self.output_dir/f"run_{result['forecast_date'].replace('-','')}_h{task['horizon']}"; out.mkdir(exist_ok=True); (out/"result.json").write_text(json.dumps(result,indent=2)); (out/"audit.json").write_text(json.dumps(audit,indent=2)); return result
    def explain(self,result,ticker):
        lines=[f"{ticker}: forecast={result['predictions'][ticker]:.8g}",f"origin={result['forecast_date']}, horizon={result['horizon']}, candidates={result['candidate_count']}","Top retrieved analogs:"]
        lines += [f"  {n['ticker']} @ endpoint {n['endpoint']}: distance={n['distance']:.6g}, weight={w:.4f}, relative_change={n['relative_change']:.6g}" for n,w in zip(result['neighbors'][:5],result['weights'][:5])]
        return "\n".join(lines)
