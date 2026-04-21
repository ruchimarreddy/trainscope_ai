# TrainScope

TrainScope is an AI-assisted experiment diagnostics tool for understanding **training dynamics, stability, overfitting, and run quality** across machine learning experiments.

I wanted to build something more useful than a simple dashboard that only plots loss curves. In research and model development, we often end up with many runs, many metrics, and a lot of trial-and-error. TrainScope is meant to act like a lightweight experiment intelligence system: it compares runs, flags instability, surfaces likely overfitting, and produces quick natural-language summaries that make training behavior easier to interpret.

## What makes it AI-powered

TrainScope now includes a lightweight **AI analyst layer**:
- a retrieval stage that builds a knowledge base from run diagnostics and metric observations
- an evidence-ranking stage using TF-IDF similarity
- an optional **local LLM** (`google/flan-t5-small` through Hugging Face Transformers) to generate grounded answers
- a fallback retrieval-plus-rules mode if the local model is unavailable

This means the app does more than hard-coded plotting. It can answer questions like:
- Why is one run better than another?
- Which run is most stable and why?
- When did overfitting begin?
- What evidence suggests generalization issues?

## What it does

- Loads one or more experiment CSV files
- Visualizes training and validation curves
- Computes diagnostic tags like:
  - stable training
  - moderately stable
  - unstable training
  - overfitting detected
  - generalization gap
  - plateaued
- Compares multiple runs side by side
- Ranks runs by validation accuracy, validation loss, or training loss
- Supports natural-language Q&A over current runs
- Shows the evidence used by the AI analyst

## Why I built it

My research is closely tied to understanding how neural networks train, when they become unstable, and how different optimization choices change learning behavior. I wanted a project that sits between theory and engineering: something practical enough to use on real runs, but also thoughtful enough to surface the kinds of signals researchers actually care about.

## Project structure

```text
trainscope/
├── app.py
├── analysis/
│   ├── __init__.py
│   ├── parser.py
│   ├── diagnostics.py
│   ├── reporting.py
│   └── ai_assistant.py
├── sample_data/
│   ├── run_adam_stable.csv
│   ├── run_sgd_overfit.csv
│   └── run_high_lr_unstable.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## Expected CSV format

Each run should contain epoch-wise metrics such as:

```text
epoch,train_loss,val_loss,train_accuracy,val_accuracy
```

## Quick start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

On Windows:

```bash
venv\Scripts\activate
streamlit run app.py
```

## How the AI layer works

1. TrainScope parses uploaded metrics.
2. It computes run diagnostics such as stability, overfitting, divergence, and validation peaks.
3. It converts those signals into textual evidence snippets.
4. It retrieves the most relevant snippets for the user's question.
5. It either:
   - sends the evidence into a small local LLM for a grounded answer, or
   - falls back to retrieval-plus-rules mode.

## Future improvements

- Read TensorBoard logs directly
- Parse config files and connect hyperparameters to training behavior
- Add anomaly scoring for exploding gradients and dead runs
- Export experiment reports as Markdown or PDF
- Add experiment-memory and multi-session run history
