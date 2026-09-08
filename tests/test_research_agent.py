from research_agent.decision import evaluate_decision
from research_agent.store import ResearchStore
from research_agent.world_model import simulate
from research_agent.orchestrator import ResearchOrchestrator

def test_world_model_demo_is_seeded_and_audited():
    a=simulate(2,8,42,["AAPL","MSFT"]); b=simulate(2,8,42,["AAPL","MSFT"])
    assert a["paths"]==b["paths"]==[]
    assert a["mode"] == "unavailable"

def test_decision_uses_paths_and_constraints():
    result=evaluate_decision([.1,.2,.3],constraints={"risk_budget":.15})
    assert result["disclaimer"]=="research simulation only"
    assert result["recommended_action"] in {x["action"] for x in result["ranked_actions"]}

def test_research_store_lifecycle(tmp_path):
    s=ResearchStore(tmp_path/"r.sqlite3"); s.save({"run_id":"r1","answer":"ok"})
    assert s.get("r1")["answer"]=="ok" and s.stop("r1")["stopped"] is True
    m=s.add_memory("paper","world model note"); assert s.search_memory("world model")[0]["id"]==m["id"]
    assert s.delete_memory(m["id"])["deleted"]==m["id"]

def test_plan_does_not_treat_words_as_tickers():
    agent=ResearchOrchestrator()
    assert {x["tool"] for x in agent.plan("Compare world model agents and volatility forecasting")}=={"srm_forecast","world_model_simulate"}
