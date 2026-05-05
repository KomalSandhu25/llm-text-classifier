# llm-text-classifier

> Production-grade multi-label text classification using fine-tuned DistilBERT — with MLflow experiment tracking, ONNX export, and a FastAPI inference server.

---

## Overview

`llm-text-classifier` is a complete, end-to-end NLP system that fine-tunes a DistilBERT model for multi-label text classification on the AG News dataset. It is designed with production patterns from the ground up: typed throughout, modular, tested, and deployable via Docker.

### Planned Architecture

```
Raw Text
   │
   ▼
TextPreprocessor          ← clean, normalise, truncate
   │
   ▼
HuggingFace Tokeniser     ← WordPiece tokens, attention mask
   │
   ▼
DistilBERT Encoder        ← contextual embeddings
   │
   ▼
Custom Classification Head ← dropout → linear → logits
   │
   ▼
BCEWithLogitsLoss          ← multi-label objective
   │
   ▼
MLflow Experiment Tracker  ← params, metrics, artefacts
   │
   ▼
ONNX Export + Benchmarking ← PyTorch → ONNX → latency comparison
   │
   ▼
FastAPI Inference Server   ← /predict endpoint
```

---

## Tech Stack

| Layer | Libraries |
|-------|-----------|
| Model | `transformers`, `torch` |
| Data | `datasets`, `nltk` |
| Training | `mlflow`, HuggingFace `Trainer` |
| Inference | `onnxruntime`, `fastapi`, `uvicorn` |
| Evaluation | `scikit-learn` |
| Config | `pydantic-settings` |

---

## Data Pipeline

The data pipeline lives in `src/data/` and handles everything from raw download through augmented `DataLoader` construction.

### Text Preprocessing (`src/data/preprocessor.py`)

`TextPreprocessor` applies a deterministic sequence of cleaning steps, each independently toggleable via `PreprocessorConfig`:

| Step | Default | Notes |
|------|---------|-------|
| Unicode normalisation (NFKC) | ✅ | Collapses full-width chars, ligatures |
| HTML unescape + tag removal | ✅ | Handles `&amp;`, `<br/>`, etc. |
| URL removal | ✅ | Strips `http://`, `https://`, `www.` |
| Special character removal | ✅ | Retains letters, digits, whitespace |
| Lowercasing | ✅ | |
| Whitespace collapsing | ✅ | Strips leading/trailing, collapses runs |
| Character truncation | optional | Word-boundary aware |

```python
from src.data.preprocessor import TextPreprocessor

prep = TextPreprocessor(max_chars=512)
prep.clean("Visit <b>https://example.com</b> NOW!!!")
# → "visit now"
```

### Dataset Loading (`src/data/loaders.py`)

`load_ag_news_dataloaders()` downloads the AG News dataset (HuggingFace Hub), splits it into train / val / test, applies preprocessing, and returns `torch.utils.data.DataLoader` instances.

- **Dataset**: AG News — 120 000 train / 7 600 test samples across 4 classes (World, Sports, Business, Sci/Tech)
- **Multi-hot labels**: Even though AG News is single-label, outputs are multi-hot `float32` tensors of shape `(4,)`, compatible with `BCEWithLogitsLoss`
- **Train/val split**: Configurable via `val_split` (default `0.1`)

```python
from src.data.loaders import load_ag_news_dataloaders

loaders = load_ag_news_dataloaders(batch_size=32, max_seq_length=128)
for batch in loaders["train"]:
    # batch["input_ids"]      → (32, 128)
    # batch["attention_mask"] → (32, 128)
    # batch["labels"]         → (32, 4)
    break
```

### Data Augmentation (`src/data/augmentation.py`)

Two augmentation strategies are available:

**Synonym Replacement** — fully implemented using NLTK WordNet:
```python
from src.data.augmentation import SynonymAugmenter

aug = SynonymAugmenter(replace_prob=0.2, seed=42)
aug.augment("The government announced a new economic policy")
# → "The authorities announced a new economic policy"
```

**Back-Translation** — interface placeholder. Subclass `BackTranslationAugmenter` and implement `_translate()` to wire up DeepL, Google Translate, or a local OPUS-MT model:
```python
class DeepLAugmenter(BackTranslationAugmenter):
    def _translate(self, text: str, target_lang: str) -> str:
        return self._client.translate_text(text, target_lang=target_lang).text
```

Augmenters can be composed with `AugmentationPipeline`:
```python
from src.data.augmentation import AugmentationPipeline, SynonymAugmenter

pipeline = AugmentationPipeline(
    augmenters=[SynonymAugmenter(replace_prob=0.15)],
    apply_prob=0.5,
)
```

---

## Quickstart

```bash
# Clone & install
git clone https://github.com/KomalSandhu25/llm-text-classifier
cd llm-text-classifier
pip install -e ".[dev]"

# Copy and edit environment variables
cp .env.example .env

# Run tests
pytest tests/ -v
```

---

## Project Status

| Day | Feature | Status |
|-----|---------|--------|
| 1 | Project scaffold, config, data interfaces | ✅ |
| 2 | Preprocessing pipeline, AG News loader, augmentation | ✅ |
| 3 | DistilBERT model + multi-label head | 🔜 |
| 4 | MLflow training loop | 🔜 |
| 5 | ONNX export + inference engine | 🔜 |
| 6 | FastAPI server + Docker | 🔜 |
