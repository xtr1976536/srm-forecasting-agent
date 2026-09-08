from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any
import uuid

@dataclass
class Evidence:
    title: str; url: str; source_type: str; snippet: str = ""; content: str = ""; reliability: str = "medium"
    accessed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    def json(self): return asdict(self)

@dataclass
class ToolEvent:
    tool: str; input: dict[str, Any]; status: str = "ok"; output: Any = None; error: str | None = None
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    def json(self): return asdict(self)

@dataclass
class ResearchRun:
    question: str; run_id: str = field(default_factory=lambda: "research_" + uuid.uuid4().hex[:12])
    plan: list[dict[str, Any]] = field(default_factory=list)
    tools: list[ToolEvent] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    answer: str = ""; warnings: list[str] = field(default_factory=list)
    def json(self): return {"run_id": self.run_id, "question": self.question, "plan": self.plan, "tools": [x.json() for x in self.tools], "evidence": [x.json() for x in self.evidence], "answer": self.answer, "warnings": self.warnings}

@dataclass
class ResearchRequest:
    question: str
    mode: str = "research"
    tickers: list[str] = field(default_factory=list)
    horizon: int = 5
    constraints: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    output_format: str = "answer"

@dataclass
class ResearchReport:
    run_id: str
    question: str
    answer: str
    evidence: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    limitations: list[str] = field(default_factory=list)
    def markdown(self):
        lines=[f"# Research Report\n\n**Question:** {self.question}\n\n## Answer\n\n{self.answer}","\n## Evidence"]
        lines += [f"- [{x.get('title','source')}]({x.get('url','')}) ({x.get('source_type','web')})" for x in self.evidence]
        lines += ["\n## Tool Trace"] + [f"- `{x.get('tool')}`: {x.get('status')}" for x in self.tools]
        lines += ["\n## Limitations"] + [f"- {x}" for x in self.limitations]
        return "\n".join(lines)
