# SRM Forecasting Agent

An auditable agent wrapper around the Shape Retrieval Model. It parses a natural-language request, selects a data adapter, validates the information boundary, retrieves geometric analog paths, explains the neighbors, and saves JSON audit artifacts.

```bash
pip install -r srm_agent/requirements.txt
python -m srm_agent.cli "forecast AAPL MSFT NVDA horizon=5 k=20 cross asset"
```

The Yahoo adapter uses daily closing prices and labels its rolling squared-return measure as an RV proxy. For publication-grade realized volatility, pass a directory containing `merged_rv_data_filled.csv` and optionally `daily_returns.csv` with `--csv`.

The deployed Streamlit entrypoint uses the full audited SRM geometry engine (`research_engine/`): five channels (curve, velocity, acceleration, curvature, global geometry), Transformer channel weights, candidate-boundary checks, and target-specific same/cross-asset retrieval. The web layer caches run metadata and presents diagnostics without changing the paper formula.

## Optional LLM tool-use prototype

The `llm_agent.py` layer is an optional research prototype for studying reliable
tool use around the unchanged SRM forecasting engine. A cloud OpenAI-compatible
LLM (configured with `LLM_BASE_URL`, `LLM_MODEL`, and `MOONSHOT_API_KEY` or
`OPENAI_API_KEY`) converts a natural-language request into the validated
`run_srm_forecast` tool call. The numerical engine remains the source of truth.

Without an API key, the same interface uses a deterministic parser so the
dashboard and tests remain reproducible. Tool traces record the parsed
arguments, validation status, backend, and errors. This prototype does not
connect to broker accounts, place real orders, or provide investment advice.

The FastAPI endpoint is `POST /api/agent/llm` with JSON `{ "request": "..." }`.
The Streamlit Agent tab exposes the same flow and reports grounded summaries
from the returned SRM values.

## Research boundary

The LLM layer only handles request parsing, tool selection, argument validation,
and result explanation. It does not alter the SRM formulas, training protocol,
data audit, or reported forecasting results. The public-data mode uses a daily
RV proxy; paper-grade experiments should use the research RV adapter.

## Web dashboard

Install `web_requirements.txt`, then run:

```bash
PYTHONPATH=. python3 -m srm_agent.run_web
```

Open `http://127.0.0.1:8000`. The dashboard supports online delayed market data, one-, five-, and 21-day forecasts, same-asset/cross-asset retrieval, nearest-path explanations, and forecast visualization. It intentionally does not connect to broker accounts or place real orders. The paper-trading layer should be added as a separate virtual-portfolio service after forecast validation.
