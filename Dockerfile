# -- Stage 1: build ----------------------------------------------------------
FROM python:3.11-slim AS build

WORKDIR /build

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# -- Stage 2: runtime --------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="Komal Sandhu <2277komal@gmail.com>"
LABEL description="LLM Text Classifier inference server"
LABEL version="1.0.0"

RUN useradd --create-home --shell /bin/bash app
WORKDIR /home/app

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY src/ src/
COPY artifacts/ artifacts/

RUN mkdir -p logs && chown -R app:app /home/app
USER app

ENV MODEL_DIR="artifacts/onnx"
ENV TOKENIZER_DIR="artifacts/tokenizer"
ENV LOG_LEVEL="info"
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen(f'http://localhost:{PORT}/health')"

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port $PORT --log-level $LOG_LEVEL --workers 1"]
