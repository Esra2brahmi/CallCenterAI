FROM python:3.12-slim

WORKDIR /app
RUN mkdir -p src/services/agent_Ai
RUN pip install --upgrade pip

COPY requirements/base.txt requirements/agent.txt  .
RUN pip install --no-cache-dir -r base.txt -r agent.txt

COPY src/services/agent_Ai/appGPT.py ./src/services/agent_Ai

EXPOSE 8000

CMD ["uvicorn", "src.services.agent_Ai.appGPT:app", "--host", "0.0.0.0", "--port", "8000"]