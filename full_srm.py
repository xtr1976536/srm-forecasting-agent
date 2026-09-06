from __future__ import annotations
import numpy as np
import torch
from research_engine.geometry import GeometryConfig
from research_engine.srm import SRMFitConfig, fit_and_predict_window, GeometryMemory
from research_engine.common import LAG

def run_full_srm(rv_frame, tickers, horizon, scope="cross_asset", criterion="qlike", neighbors=20):
    rv=np.asarray(rv_frame.loc[:,tickers],dtype=float)
    n=len(rv)
    validation_end=n-horizon
    validation_start=validation_end-200
    train_end=validation_start-horizon
    train_start=train_end-600
    if train_start < LAG:
        required=LAG+600+horizon+200+horizon
        raise ValueError(f"full SRM requires at least {required} daily observations; received {n}")
    train=np.arange(train_start,train_end,dtype=int)
    validation=np.arange(validation_start,validation_end,dtype=int)
    test=np.asarray([n],dtype=int)
    cfg=GeometryConfig(length=22,neighbors=int(neighbors),seed=42)
    fit=SRMFitConfig(train_stride=5,training_neighbors=100,learning_rate=1e-3,epochs=100,patience=10,seed=42)
    device=torch.device("cpu")
    pred, diagnostics, audit=fit_and_predict_window(rv,train,validation,test,horizon,criterion,cfg,fit,device,scope=scope)
    values={ticker:float(pred[0,i]) for i,ticker in enumerate(tickers)}
    audit["online_window"]={"train_start":int(train[0]),"train_end":int(train[-1]),"validation_start":int(validation[0]),"validation_end":int(validation[-1]),"forecast_origin":int(n),"purge":int(horizon)}
    return {"predictions":values,"diagnostics":diagnostics,"audit":audit,"full_engine":True,"criterion":criterion}
