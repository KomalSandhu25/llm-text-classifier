# 📰 LLM Text Classifier

> Fine-tuned **DistilBERT** for multi-class text classification with **MLflow** experiment tracking, **ONNX** export for fast inference, and a **FastAPI** serving layer.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-orange)](https://huggingface.co/transformers/)
[![MLflow](https://img.shields.io/badge/MLflow-2.16-blue)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Planned Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  LLM Text Classifier                      │
│                                                           │
│  ┌─────────────┐    ┌──────────────┐   ┌──────────────┐  │
│  │  Data Layer │    │  Model Layer │   │ Training     │  │
│  │             │    │              │   │              │  │
│  │ AG News     │───▶│ DistilBERT   │──▶│ HF Trainer   │  │
│  │ Preprocessor│    │ + Custom     │   │ + MLflow     │  │
│  │ DataLoader  │    │   Head       │   │   Tracking   │  │
│  └─────────────┘    └──────────────┘   └──────┬───────┘  │
│                                                │          │
│  ┌──────────────┐    ┌─────────────────────────▼───────┐  │
│  │  FastAPI     │    │         ONNX Export             │  │
│  │  /predict    │◀───│  PyTorch → ONNX → Optimised     │  │
│  │  /health     │    │  Runtime (3-5× faster)          │  │
│  └──────────────┘    └─────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

## Project Status

🚧 **Under active development** — building incrementally.

| Component | Status |
|---|---|
| Project scaffold & config | ✅ Done |
| Data pipeline | 🔜 Next |
| DistilBERT fine-tuning | 🔜 Planned |
| MLflow tracking | 🔜 Planned |
| ONNX export | 🔜 Planned |
| FastAPI server | 🔜 Planned |
| Docker deployment | 🔜 Planned |

## Dataset

[AG News](https://huggingface.co/datasets/ag_news) — 120K news articles across 4 classes:
- 🌍 World
- ⚽ Sports
- 💼 Business
- 🔬 Science/Technology

## Quick Start *(once complete)*

```bash
git clone https://github.com/KomalSandhu25/llm-text-classifier
cd llm-text-classifier
pip install -r requirements.txt
cp .env.example .env

# Train
python scripts/train.py

# Export to ONNX
python scripts/export.py --checkpoint models/best_checkpoint

# Serve
uvicorn src.api.main:app --reload
```

---

*Part of an ML engineering portfolio — [github.com/KomalSandhu25](https://github.com/KomalSandhu25)*
