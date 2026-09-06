# Cloud Deployment

## Recommended path

1. Create a GitHub repository and upload the project root, including `srm_agent/` and `experiments/`.
2. Connect the repository to Render.
3. Choose the included `srm_agent/render.yaml` blueprint, or create a Docker web service using `srm_agent/Dockerfile`.
4. Set the service start command to:

```bash
uvicorn srm_agent.web_app:app --host 0.0.0.0 --port $PORT
```

5. Open the generated HTTPS URL.

## Important data note

The public demo uses Yahoo Finance daily data and labels the resulting volatility measure as a daily-data RV proxy. Do not describe this as high-frequency realized volatility. For a production version, replace the Yahoo adapter with a licensed intraday-data provider and store credentials as cloud environment variables.

## Scope

This deployment provides forecasts, model comparisons, path explanations, and research-only simulation. It does not connect to a broker and does not place real orders.
