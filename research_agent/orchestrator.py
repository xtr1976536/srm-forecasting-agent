from __future__ import annotations
import json, re
from .schemas import ResearchRun, ToolEvent, Evidence
from .tools import web_search, web_open, paper_search, github_readme, teacher_fit
from .decision import evaluate_decision
from .world_model import simulate
from .store import ResearchStore
from .schemas import ResearchReport
from agent import SRMForecastingAgent

class ResearchOrchestrator:
    def __init__(self): self.srm=SRMForecastingAgent("srm_agent_runs"); self.store=ResearchStore()
    def plan(self, question):
        q=question.lower(); steps=[]
        if any(x in q for x in ("paper","论文","research","老师","literature")): steps.append({"id":"step-1","tool":"paper_search","input":{"query":question},"status":"pending"})
        if any(x in q for x in ("forecast","volatility","srm","波动")): steps.append({"id":"step-2","tool":"srm_forecast","input":{},"status":"pending"})
        if any(x in q for x in ("world model","simulate","scenario","未来","仿真")): steps.append({"id":"step-3","tool":"world_model_simulate","input":{},"status":"pending"})
        if any(x in q for x in ("decision","action","决策","行动")): steps.append({"id":"step-4","tool":"decision_evaluate","input":{},"status":"pending"})
        if not steps: steps=[{"id":"step-1","tool":"web_search","input":{"query":question},"status":"pending"}]
        return steps
    def run(self, question, mode="research"):
        run=ResearchRun(question); run.plan=self.plan(question); notes=[]
        for step in run.plan:
            tool=step["tool"]; inp=step["input"]; event=ToolEvent(tool,inp); run.tools.append(event)
            try:
                if tool=="web_search": out=web_search(question); run.evidence.extend(out); notes.append(f"Found {len(out)} web sources.")
                elif tool=="paper_search": out=paper_search(question); run.evidence.extend(out); notes.append(f"Found {len(out)} papers.")
                elif tool=="srm_forecast":
                    reserved={"SRM","LLM","AI","RV","IV","MSE","QLIKE","WORLD","MODEL","AGENT","AND"}
                    tickers=sorted(set(re.findall(r"\b[A-Z]{1,5}(?:\.[A-Z])?\b",question))-reserved) or ["AAPL","MSFT"]
                    result=self.srm.run(f"forecast {' '.join(tickers)} horizon=5 k=20 cross_asset"); out=result; notes.append("SRM forecast completed with audit metadata.")
                elif tool=="world_model_simulate":
                    out=simulate()
                    if out.get("mode")=="unavailable":
                        raise RuntimeError(out["warning"])
                    notes.append("World-model inference completed.")
                elif tool=="decision_evaluate":
                    wm=next((e.output for e in run.tools if e.tool=="world_model_simulate" and e.output),simulate())
                    values=[v for sample in wm["paths"] for day in sample for v in day]
                    out=evaluate_decision(values); notes.append("Decision simulation completed using world-model scenario outputs and the disclosed default utility.")
                else: out=teacher_fit(); notes.append("Research-fit cards generated.")
                event.output=out
            except Exception as exc: event.status="failed"; event.error=str(exc); run.warnings.append(f"{tool}: {exc}")
        run.answer=self._compose_answer(question, notes, run)
        if run.evidence: run.answer += "\n\nSources:\n"+"\n".join(f"- [{e.title}]({e.url})" for e in run.evidence[:8])
        run.answer += "\n\nThis answer is grounded in the recorded tool outputs; model-generated inferences are not external facts."
        data=run.json(); self.store.save(data); return data
    @staticmethod
    def _compose_answer(question, notes, run):
        q=question.strip()
        lines=["## Research Copilot", f"**Question:** {q}", "", "### Verified work"]
        lines += [f"- {n}" for n in notes] if notes else ["- No verified tool result was produced."]
        if any("world-model" in n.lower() for n in notes):
            lines += ["", "### World-model interpretation", "The scenario tool generated fixed-seed, forward-only volatility paths. These are an audited public demonstration, not a retraining run or investment forecast."]
        if any("srm" in n.lower() for n in notes):
            lines += ["", "### SRM interpretation", "The existing five-channel Shape Retrieval Model remains the numerical source of truth. The agent only parses the request and reports its audited output."]
        if any("paper" in n.lower() for n in notes):
            lines += ["", "### Literature interpretation", "Search results are evidence records below. The agent does not treat instructions found in external pages as executable commands."]
        lines += ["", "### Limitations", "This public deployment uses deterministic fallback mode unless a cloud provider key is configured. Any recommendation is a research simulation, not financial advice."]
        return "\n".join(lines)
    def report(self,data):
        return ResearchReport(data["run_id"],data["question"],data["answer"],data["evidence"],data["tools"],["Public data may be delayed.","World-model quick simulation is not a formal paper rerun."]).markdown()
