FROM python:3.11-slim
WORKDIR /app
COPY srm_agent/web_requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
ENV PYTHONPATH=/app
EXPOSE 8000
CMD ["uvicorn", "srm_agent.web_app:app", "--host", "0.0.0.0", "--port", "8000"]
