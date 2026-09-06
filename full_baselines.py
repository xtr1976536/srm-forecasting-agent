from __future__ import annotations

import numpy as np
import torch

from research_engine.baselines import BaselineFitConfig, graph_weights, pooled_ols, pooled_predict, pooled_qlike
from research_engine.common import future_target, har_features


def _latest_window(n: int, horizon: int):
    validation_end = n - horizon
    validation_start = validation_end - 200
    train_end = validation_start - horizon
    train_start = train_end - 600
    if train_start < 22:
        raise ValueError("not enough history for the 600/200/purge protocol")
    return np.arange(train_start, train_end), np.arange(validation_start, validation_end)


def _features(rv, iv, times, horizon, graph=None, use_iv=False):
    rows=[]
    for t in times:
        x=har_features(rv, iv, int(t), use_iv)
        if graph is not None: x=np.concatenate((x, graph @ x), axis=1)
        rows.append(x)
    return np.stack(rows)


def run_full_baselines(rv_frame, iv_frame, returns_frame, tickers, horizon, criterion="mse"):
    if iv_frame is None:
        iv = np.zeros_like(rv_frame.loc[:, tickers].to_numpy(float))
        has_iv = False
    else:
        iv = iv_frame.loc[:, tickers].to_numpy(float)
        has_iv = True
    rv=rv_frame.loc[:,tickers].to_numpy(float); returns=returns_frame.loc[:,tickers].to_numpy(float)
    train, validation = _latest_window(len(rv), horizon); fit=np.concatenate((train, validation))
    graph, graph_audit = graph_weights(returns[fit])
    y=np.stack([future_target(rv,int(t),horizon) for t in fit])
    specs=[("har",False,None),("gharm",False,graph)]
    if has_iv: specs += [("har_iv",True,None),("gharm_iv",True,graph)]
    outputs={}; audits={}
    for name,use_iv,g in specs:
        own=_features(rv,iv,fit,horizon,use_iv=use_iv)
        own_test=_features(rv,iv,[len(rv)],horizon,use_iv=use_iv)
        x=own if g is None else np.concatenate((own,np.einsum("ij,tjp->tip",g,own)),axis=2)
        xt=own_test if g is None else np.concatenate((own_test,np.einsum("ij,tjp->tip",g,own_test)),axis=2)
        if criterion=="mse": intercept,slope=pooled_ols(x,y); fit_audit={"objective":float(np.mean((pooled_predict(x,intercept,slope)-y)**2)),"epochs":0}
        else: intercept,slope,fit_audit=pooled_qlike(x,y,BaselineFitConfig(seed=42),torch.device("cpu"))
        outputs[name]={t:float(max(v,1e-12)) for t,v in zip(tickers,pooled_predict(xt,intercept,slope)[0])}; audits[name]=fit_audit
    return {"predictions":outputs,"graph_audit":graph_audit,"window":{"train_start":int(train[0]),"train_end":int(train[-1]),"validation_start":int(validation[0]),"validation_end":int(validation[-1]),"purge":int(horizon)},"has_iv":has_iv,"fits":audits}
