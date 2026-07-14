FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY api ./api
COPY config ./config
COPY alembic.ini ./
COPY alembic ./alembic
RUN python -m pip install --no-cache-dir -U pip && python -m pip install --no-cache-dir -e ".[api,models]"

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
