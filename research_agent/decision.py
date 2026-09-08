from __future__ import annotations
import math

DEFAULT_ACTIONS=["maintain exposure","reduce exposure","increase protection","run additional data collection","defer decision"]
def evaluate_decision(paths, actions=None, constraints=None, transaction_cost=0.0):
    actions=actions or DEFAULT_ACTIONS; constraints=constraints or {}; paths=[float(x) for x in paths]
    if not paths: raise ValueError("at least one simulated path value is required")
    mean=sum(paths)/len(paths); q90=sorted(paths)[max(0,int(.9*len(paths))-1)]
    risk_budget=constraints.get("risk_budget")
    ranked=[]
    for action in actions:
        multiplier={"maintain exposure":1.0,"reduce exposure":.65,"increase protection":.35,"run additional data collection":.9,"defer decision":.75}.get(action,1.0)
        score=mean*multiplier + transaction_cost
        feasible=risk_budget is None or q90*multiplier <= float(risk_budget)
        ranked.append({"action":action,"expected_risk":score,"tail_risk":q90*multiplier,"feasible":feasible})
    feasible=[x for x in ranked if x["feasible"]]
    best=min(feasible or ranked,key=lambda x:x["expected_risk"])
    return {"ranked_actions":sorted(ranked,key=lambda x:x["expected_risk"]),"scenario_metrics":{"mean":mean,"q90":q90},"constraint_violations":[x["action"] for x in ranked if not x["feasible"]],"recommended_action":best["action"],"explanation":"Research simulation only; ranking is grounded in supplied path values and constraints.","disclaimer":"research simulation only"}
