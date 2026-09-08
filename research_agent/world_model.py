from __future__ import annotations
import csv, json, os
from pathlib import Path

def _repo():
    configured=os.getenv("WORLD_MODEL_REPO")
    candidates=[Path(configured)] if configured else []
    candidates += [Path(__file__).resolve().parents[2]/"volatility-world-model",Path(__file__).resolve().parents[1]/"world_model_assets"]
    return next((p for p in candidates if p and p.exists()),None)

def simulate(horizon=5,n_samples=100,seed=20260721,assets=None,public_shock=True):
    horizon=int(horizon); n_samples=min(max(int(n_samples),8),2000); assets=assets or ["AAPL","MSFT","JPM"]
    root=_repo(); audit={}; summaries=[]
    if root:
        ap=root/"results/dow30_serial/final_audit.json"; sp=root/"results/dow30_serial/summary.csv"
        if ap.exists(): audit=json.loads(ap.read_text())
        if sp.exists(): summaries=list(csv.DictReader(sp.open()))
    checkpoint=next(root.glob("**/*.pt"),None) if root else None
    # Public web requests must not present synthetic paths as paper-model inference.
    return {"mode":"unavailable","assets":assets,"horizon":horizon,"n_samples":n_samples,"seed":int(seed),"public_shock":public_shock,"paths":[],"summary":{},"formal_audit":audit,"reported_summary":summaries[:20],"checkpoint_configured":False,"warning":"World-model inference is not connected. No simulated paths were produced."}
