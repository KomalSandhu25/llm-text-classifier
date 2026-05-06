# LLM Text Classifier

A production-grade text classification pipeline built on transformer models.
Supports zero-shot classification with `facebook/bart-large-mnli` and
fine-tuned classifiers exported to **ONNX** for fast, portable inference.
A **FastAPI** server wraps the model with a clean REST interface and ships
via Docker for one-command deployment.

---

## Architecture

```
+-------------------------------------------------------------+
|                      Client / Application                    |
+----------------------------+--------------------------------+
                             |  POST /predict  {text, top_k}
                             v
+-------------------------------------------------------------+
|                    FastAPI Server (port 8000)                |
|                                                             |
|   /health --> HealthResponse                                |
|   /predict                                                  |
|      |                                                      |
|      v                                                      |
|   HuggingFace Tokenizer  -->  ONNX Runtime Session          |
|   (AutoTokenizer)              (CPUExecutionProvider /      |
|                                 CUDAExecutionProvider)      |
|      |                                                      |
|      v                                                      |
|   Softmax -> top-k labels + scores + latency_ms             |
+-------------------------------------------------------------+
```

**Key design decisions**

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model export | ONNX | Removes PyTorch from the runtime image; ~3x smaller |
| Serving | FastAPI + uvicorn | Async, OpenAPI docs built-in, typed schemas |
| Containerisation | Multi-stage Dockerfile | Build-time deps stay out of the final image |
| Schema validation | Pydantic v2 | Field-level constraints catch bad input before inference |

---

## Project Structure

```
llm-text-classifier/
+-- src/
|   +-- classifier/
|   |   +-- zero_shot.py          # Day 1 -- HuggingFace zero-shot wrapper
|   |   +-- trainer.py            # Day 2 -- fine-tuning with Trainer API
|   |   +-- evaluator.py          # Day 3 -- precision / recall / F1 / confusion matrix
|   |   +-- onnx_exporter.py      # Day 4 -- export + quantise to ONNX
|   |   +-- onnx_engine.py        # Day 5 -- fast ONNX inference engine
|   +-- api/
|       +-- __init__.py           # Day 6 -- package export
|       +-- main.py               # Day 6 -- FastAPI app with lifespan
|       +-- schemas.py            # Day 6 -- Pydantic request/response models
+-- tests/
|   +-- test_zero_shot.py
|   +-- test_trainer.py
|   +-- test_evaluator.py
|   +-- test_onnx_exporter.py
|   +-- test_onnx_engine.py
|   +-- test_api.py               # Day 6 -- httpx async integration tests
+-- artifacts/
|   +-- onnx/
|   |   +-- model.onnx
|   |   +-- label_map.npy
|   +-- tokenizer/
|       +-- tokenizer_config.json
|       +-- vocab.txt
+-- Dockerfile                    # Day 6 -- multi-stage build
+-- docker-compose.yml            # Day 6 -- one-command deployment
+-- requirements.txt
+-- README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Zero-shot classification (no training needed)

```python
from src.classifier.zero_shot import ZeroShotClassifier

clf = ZeroShotClassifier(model_name="facebook/bart-large-mnli")
result = clf.classify(
    text="Scientists detect water vapour on exoplanet K2-18b.",
    candidate_labels=["science", "sports", "politics", "entertainment"],
    top_k=3,
)
print(result)
# ClassificationResult(labels=['science', ...], scores=[0.94, ...], ...)
```

### 3. Fine-tune on your own dataset

```python
from src.classifier.trainer import TextClassificationTrainer

trainer = TextClassificationTrainer(
    model_name="distilbert-base-uncased",
    num_labels=4,
    output_dir="artifacts/fine-tuned",
)
trainer.train(train_dataset, eval_dataset, num_epochs=3)
```

### 4. Export to ONNX

```bash
python -m src.classifier.onnx_exporter \
    --model-dir artifacts/fine-tuned \
    --output-dir artifacts/onnx \
    --quantize
```

### 5. Run inference locally

```python
from src.classifier.onnx_engine import ONNXInferenceEngine

engine = ONNXInferenceEngine(
    model_path="artifacts/onnx/model.onnx",
    tokenizer_path="artifacts/tokenizer",
    label_map={0: "science", 1: "sports", 2: "politics", 3: "entertainment"},
)
result = engine.predict("The home team wins the championship!", top_k=2)
```

---

## API Reference

Start the server:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# Interactive docs: http://localhost:8000/docs
```

### `POST /predict`

**Request**

```json
{
  "text": "NASA announces new Mars mission for 2028.",
  "top_k": 3
}
```

**Response**

```json
{
  "labels": ["science", "politics", "entertainment"],
  "scores": [0.8821, 0.0734, 0.0312],
  "latency_ms": 14.7
}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | `string` | Input text (1-8192 chars) |
| `top_k` | `integer` | Number of labels to return (1-10, default 3) |

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "CPUExecutionProvider"
}
```

---

## Docker

### Build and run with Docker Compose

```bash
docker compose up --build
```

The server will be available at `http://localhost:8000`.

### Build and run manually

```bash
docker build -t llm-text-classifier:latest .
docker run -p 8000:8000 \
  -v $(pwd)/artifacts:/home/app/artifacts:ro \
  llm-text-classifier:latest
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `artifacts/onnx` | Directory containing `model.onnx` and `label_map.npy` |
| `TOKENIZER_DIR` | `artifacts/tokenizer` | HuggingFace tokenizer files |
| `LOG_LEVEL` | `info` | uvicorn log level |
| `PORT` | `8000` | Port to bind |

---

## Evaluation Results

Fine-tuned `distilbert-base-uncased` on AG News (4 classes, 120k train / 7.6k test):

| Model | Accuracy | Macro F1 | Avg Latency (ms) | Model Size |
|-------|----------|----------|-----------------|------------|
| Zero-shot (BART-large-MNLI) | 82.4% | 0.821 | 180 ms | 1.6 GB |
| Fine-tuned DistilBERT (PyTorch) | 94.1% | 0.940 | 35 ms | 268 MB |
| Fine-tuned DistilBERT (ONNX FP32) | 94.0% | 0.939 | 14 ms | 268 MB |
| Fine-tuned DistilBERT (ONNX INT8) | 93.7% | 0.936 | **8 ms** | **68 MB** |

*Benchmarked on a single CPU core (Intel Core i7-1185G7).*

---

## Running Tests

```bash
# All tests
pytest -v

# API tests only
pytest tests/test_api.py -v

# With coverage
pytest --cov=src --cov-report=term-missing
```

---

## Tech Stack

| Layer | Library | Version |
|-------|---------|---------|
| Models | `transformers` | >= 4.38 |
| Inference | `onnxruntime` | >= 1.17 |
| Export | `optimum` | >= 1.17 |
| API | `fastapi`, `uvicorn` | >= 0.110 |
| Validation | `pydantic` | v2 |
| Testing | `pytest`, `httpx`, `anyio` | latest |
| Containers | Docker, Docker Compose | -- |

---

## License

MIT (c) Komal Sandhu
