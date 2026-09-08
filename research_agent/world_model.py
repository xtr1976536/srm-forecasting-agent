from __future__ import annotations
import csv, json, os
from pathlib import Path
import numpy as np

def _repo():
    configured=os.getenv("WORLD_MODEL_REPO")
    candidates=[Path(configured)] if configured else []
    candidates += [Path(__file__).resolve().parents[2]/"volatility-world-model",Path(__file__).resolve().parents[1]/"world_model_assets"]
    return next((p for p in candidates if p and p.exists()),None)

def simulate(horizon=5,n_samples=100,seed=20260721,assets=None,public_shock=True):
    horizon=int(horizon); n_samples=min(max(int(n_samples),8),2000); rng=np.random.default_rng(int(seed)); assets=assets or ["AAPL","MSFT","JPM"]
    root=_repo(); audit={}; summaries=[]
    if root:
        ap=root/"results/dow30_serial/final_audit.json"; sp=root/"results/dow30_serial/summary.csv"
        if ap.exists(): audit=json.loads(ap.read_text())
        if sp.exists(): summaries=list(csv.DictReader(sp.open()))
    base=np.array([0.18+0.025*i for i in range(len(assets))]); common=rng.normal(0,.025,(n_samples,horizon,1)) if public_shock else 0
    idio=rng.normal(0,.035,(n_samples,horizon,len(assets))); paths=np.maximum(.0001,base[None,None,:]+common+idio).tolist()
    arr=np.asarray(paths)
    return {"mode":"audited-results-driven-demo","assets":assets,"horizon":horizon,"n_samples":n_samples,"seed":int(seed),"public_shock":public_shock,"paths":paths[:50],"summary":{"mean":arr.mean((0,1)).tolist(),"q10":np.quantile(arr,.1,axis=(0,1)).tolist(),"q90":np.quantile(arr,.9,axis=(0,1)).tolist()},"formal_audit":audit,"reported_summary":summaries[:20],"future_observations_used":False,"checkpoint_configured":False,"warning":"Fast public simulation; this does not retrain or reproduce the formal paper experiment."}
