from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def anonymized_snapshot(result: dict) -> dict:
    """Return only safe aggregate information suitable for public storage."""
    audit = result.get("data_audit", {})
    return {
        "run_id": result.get("run_id"),
        "date": result.get("forecast_date"),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "tickers": sorted(list(result.get("predictions", {}).keys())),
        "horizon": result.get("horizon"),
        "criterion": result.get("criterion") or result.get("task", {}).get("criterion"),
        "model": result.get("model"),
        "data_mode": "daily_rv_proxy" if audit.get("is_daily_rv_proxy") else "user_research_data",
        "predictions_summary": {
            "min": min(result.get("predictions", {}).values()) if result.get("predictions") else None,
            "max": max(result.get("predictions", {}).values()) if result.get("predictions") else None,
            "mean": sum(result.get("predictions", {}).values()) / len(result["predictions"]) if result.get("predictions") else None,
        },
        "audit_status": "passed" if not result.get("warnings") and not result.get("full_engine_warning") else "warning",
        "warnings": list(result.get("warnings", [])),
    }


def write_snapshot(result: dict, output_dir: str | Path = "public_snapshots") -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    snapshot = anonymized_snapshot(result)
    out = path / f"{snapshot['date']}_{snapshot['run_id']}.json"
    out.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return out
