FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-deps .

RUN chgrp -R 0 /app && chmod -R g=u /app

USER 1001
EXPOSE 8080

CMD ["sh", "-c", "uvicorn att_agent.api:app --host 0.0.0.0 --port ${PORT:-8080}"]

