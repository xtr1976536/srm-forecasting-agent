from __future__ import annotations
import numpy as np
import torch
from research_engine.geometry import GeometryConfig
from research_engine.srm import SRMFitConfig, fit_and_predict_window, GeometryMemory
from research_engine.common import LAG

def run_full_srm(rv_frame, tickers, horizon, scope="cross_asset", criterion="qlike"):
    rv=np.asarray(rv_frame.loc[:,tickers],dtype=float)
    n=len(rv)
    train_start=LAG
    train_end=train_start+600
    val_end=train_end+horizon+200
    if val_end+horizon >= n:
        raise ValueError(f"full SRM requires at least {val_end+horizon+1} daily observations; received {n}")
    train=np.arange(train_start,train_end,dtype=int)
    validation=np.arange(train_end+horizon,val_end+horizon,dtype=int)
    test=np.asarray([n],dtype=int)
    cfg=GeometryConfig(length=22,neighbors=20,seed=42)
    fit=SRMFitConfig(train_stride=5,training_neighbors=100,learning_rate=1e-3,epochs=100,patience=10,seed=42)
    device=torch.device("cpu")
    pred, diagnostics, audit=fit_and_predict_window(rv,train,validation,test,horizon,criterion,cfg,fit,device,scope=scope)
    values={ticker:float(pred[0,i]) for i,ticker in enumerate(tickers)}
    return {"predictions":values,"diagnostics":diagnostics,"audit":audit,"full_engine":True,"criterion":criterion}
