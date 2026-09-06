# Free Deployment: Streamlit Community Cloud

1. Open https://share.streamlit.io/ and sign in with the GitHub account that owns `xtr1976536/srm-forecasting-agent`.
2. Click **Create app**.
3. Select repository `xtr1976536/srm-forecasting-agent`, branch `main`, and file path `streamlit_app.py`.
4. Set Python dependencies to `srm_agent/streamlit_requirements.txt` if the UI asks for a requirements file, or keep the file in the repository root by copying its contents to `requirements.txt`.
5. Deploy. Streamlit will provide a public `*.streamlit.app` URL.

The free service may sleep after inactivity. Session-level paper-account state is intentionally non-persistent; no real orders or broker keys are used.

After each GitHub push, Streamlit Community Cloud automatically rebuilds the app. Use the sidebar's Continuous refresh switch for periodic polling; the minimum interval is deliberately conservative to avoid excessive Yahoo Finance requests.
