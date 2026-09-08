"""Optional cloud LLM interface for the auditable SRM forecasting agent.

The numerical SRM engine remains the source of truth. This module only turns
natural-language requests into a validated tool call and explains results.
It uses an OpenAI-compatible endpoint when configured and a deterministic
fallback otherwise, so the research app remains usable without a secret.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Callable
from urllib.request import Request, urlopen

from agent import SRMForecastingAgent

TOOL_SCHEMAS = [{
    "type": "function",
    "function": {
        "name": "run_srm_forecast",
        "description": "Run the existing five-channel SRM forecast after validating the request.",
        "parameters": {
            "type": "object",
            "properties": {
                "tickers": {"type": "array", "items": {"type": "string"}},
                "horizon": {"type": "integer", "enum": [1, 5, 21]},
                "k": {"type": "integer", "minimum": 5, "maximum": 100},
                "scope": {"type": "string", "enum": ["cross_asset", "same_asset"]},
                "criterion": {"type": "string", "enum": ["qlike", "mse"]},
            },
            "required": ["tickers", "horizon", "k", "scope", "criterion"],
            "additionalProperties": False,
        },
    }
}]

@dataclass
class ToolTrace:
    name: str
    arguments: dict[str, Any]
    status: str
    source: str
    error: str | None = None

class LLMForecastingAgent:
    def __init__(self, numerical_agent: SRMForecastingAgent | None = None):
        self.numerical_agent = numerical_agent or SRMForecastingAgent("srm_agent_runs")
        self.traces: list[ToolTrace] = []

    @staticmethod
    def _fallback_parse(text: str) -> dict[str, Any]:
        tickers = sorted(set(re.findall(r"\b[A-Z]{1,5}(?:\.[A-Z])?\b", text)))
        low = text.lower()
        hm = re.search(r"(?:horizon|h|未来)\s*[=:]?\s*(1|5|21)", low)
        km = re.search(r"(?:k|neighbor|近邻)\s*[=:]?\s*(\d+)", low)
        return {"tickers": tickers, "horizon": int(hm.group(1)) if hm else 1,
                "k": min(max(int(km.group(1)) if km else 20, 5), 100),
                "scope": "same_asset" if ("same asset" in low or "同资产" in low) else "cross_asset",
                "criterion": "mse" if re.search(r"criterion\s*[=:]?\s*mse", low) else "qlike"}

    def _cloud_parse(self, text: str) -> tuple[dict[str, Any], str] | None:
        key = os.getenv("MOONSHOT_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not key:
            return None
        base = os.getenv("LLM_BASE_URL", "https://api.moonshot.cn/v1").rstrip("/")
        model = os.getenv("LLM_MODEL", "kimi-k2.5")
        payload = {"model": model, "temperature": 0, "messages": [
            {"role": "system", "content": "Extract a valid run_srm_forecast tool call. Never invent tickers. Use horizon 1, 5, or 21; k 5-100; scope cross_asset or same_asset; criterion qlike or mse."},
            {"role": "user", "content": text}], "tools": TOOL_SCHEMAS, "tool_choice": {"type": "function", "function": {"name": "run_srm_forecast"}}}
        req = Request(base + "/chat/completions", data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        with urlopen(req, timeout=30) as response:
            data = json.loads(response.read())
        call = data["choices"][0]["message"].get("tool_calls", [])[0]
        return json.loads(call["function"]["arguments"]), "cloud:" + model

    @staticmethod
    def _validate(args: dict[str, Any]) -> dict[str, Any]:
        tickers = [str(t).upper() for t in args.get("tickers", []) if re.fullmatch(r"[A-Z]{1,5}(?:\.[A-Z])?", str(t).upper())]
        if not tickers: raise ValueError("No valid ticker was found")
        horizon = int(args.get("horizon", 1)); k = int(args.get("k", 20))
        if horizon not in (1, 5, 21): raise ValueError("horizon must be 1, 5, or 21")
        if not 5 <= k <= 100: raise ValueError("k must be between 5 and 100")
        scope = args.get("scope", "cross_asset"); criterion = args.get("criterion", "qlike")
        if scope not in ("cross_asset", "same_asset") or criterion not in ("qlike", "mse"): raise ValueError("invalid scope or criterion")
        return {"tickers": tickers, "horizon": horizon, "k": k, "scope": scope, "criterion": criterion}

    def run(self, text: str, csv_path: str | None = None) -> dict[str, Any]:
        self.traces = []
        try:
            parsed = self._cloud_parse(text)
        except Exception as exc:
            parsed = None
            cloud_error = str(exc)
        source = parsed[1] if parsed else "deterministic-fallback"
        args = self._validate(parsed[0] if parsed else self._fallback_parse(text))
        self.traces.append(ToolTrace("run_srm_forecast", args, "validated", source))
        request = f"forecast {' '.join(args['tickers'])} horizon={args['horizon']} k={args['k']} {args['scope']} criterion={args['criterion']}"
        try:
            result = self.numerical_agent.run(request, csv_path)
            result["llm_agent"] = {"backend": source, "tool_traces": [asdict(t) for t in self.traces], "cloud_error": locals().get("cloud_error")}
            return result
        except Exception as exc:
            self.traces[-1].status = "failed"; self.traces[-1].error = str(exc)
            raise

    def explain(self, result: dict[str, Any]) -> str:
        task = result.get("task", {})
        forecasts = ", ".join(f"{t}: {v:.6g}" for t, v in result.get("predictions", {}).items())
        return (f"SRM forecast for {', '.join(task.get('tickers', []))} over {task.get('horizon')} days: {forecasts}. "
                f"The result uses {task.get('k')} retrieved candidates under {task.get('scope')} retrieval. "
                "Values are volatility magnitudes, not price-direction predictions or investment advice.")
