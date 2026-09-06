import uvicorn

if __name__ == "__main__":
    uvicorn.run("srm_agent.web_app:app", host="127.0.0.1", port=8000, reload=False)
