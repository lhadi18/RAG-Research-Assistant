FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY evals ./evals
ARG INSTALL_OBSERVABILITY=false
RUN pip install --no-cache-dir . && if [ "$INSTALL_OBSERVABILITY" = "true" ]; then pip install --no-cache-dir '.[observability]'; fi
EXPOSE 8000 8501
CMD ["uvicorn", "rag_assistant.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
