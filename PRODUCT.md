# Product boundaries and operating model

## Two clocks

The platform separates two different information clocks.

1. **Daily forecast clock.** HARM, GHARM and SRM are re-estimated after a completed daily observation becomes available. The output is the next-day or future 5/21-day average volatility forecast.
2. **Intraday observation clock.** Minute bars update the current session's provisional cumulative realized variance. This monitor does not retrain SRM and does not claim a minute-ahead SRM forecast.

## User workflows

- **Forecast Monitor:** choose assets, model, horizon, criterion, K and retrieval scope; compare daily volatility forecasts.
- **Intraday Risk:** inspect price and provisional current-session realized volatility from a public or owner-configured provider.
- **Model Lab:** inspect model availability, configuration and research backtest provenance.
- **Analog Explorer:** inspect selected paths, channel weights and candidate diagnostics.
- **Forecast Agent:** translate natural-language analysis requests into audited model calls.
- **Paper Trading:** apply volatility forecasts to risk scaling and record simulated orders. Volatility does not determine price direction.
- **Audit:** inspect source, timestamp, proxy status and information-set boundaries.

## Data policy

- Yahoo and Stooq are best-effort public sources and are labelled as daily RV proxies.
- Alpha Vantage intraday access is optional and requires an owner-managed Streamlit secret.
- User uploads remain session-scoped and must contain positive RV data with unique increasing dates.
- No broker credentials, real accounts or real orders are supported.

## Design references

The information architecture follows established volatility platforms: daily model updates, horizon forecasts, historical/current risk comparisons, explicit data-feed status and model documentation. SRM adds geometric analog retrieval and channel diagnostics as its differentiating feature.
