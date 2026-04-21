from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from analysis.diagnostics import RunDiagnostics, diagnose_run
from analysis.parser import RunData
from analysis.reporting import build_run_report, build_comparison_report


@dataclass
class AIResponse:
    answer: str
    evidence: list[str]
    mode: str


@lru_cache(maxsize=1)
def _load_generator():
    try:
        from transformers import pipeline

        return pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            max_new_tokens=220,
            truncation=True,
        )
    except Exception:
        return None


def build_knowledge_base(runs: List[RunData], comparison_df: pd.DataFrame, ranking_metric: str) -> list[dict]:
    docs: list[dict] = []
    docs.append(
        {
            "title": "Global comparison summary",
            "text": build_comparison_report(comparison_df, ranking_metric),
        }
    )

    for run in runs:
        diag = diagnose_run(run.data)
        docs.append({"title": f"Run summary: {run.name}", "text": build_run_report(run.name, diag)})
        docs.extend(_build_metric_observations(run, diag))
    return docs


def _build_metric_observations(run: RunData, diag: RunDiagnostics) -> list[dict]:
    df = run.data
    docs: list[dict] = []

    if "val_accuracy" in df.columns and df["val_accuracy"].notna().any():
        best_idx = int(df["val_accuracy"].idxmax())
        best_epoch = int(df.loc[best_idx, "epoch"])
        best_val_acc = float(df.loc[best_idx, "val_accuracy"])
        docs.append(
            {
                "title": f"Validation peak: {run.name}",
                "text": f"{run.name} reaches its best validation accuracy of {best_val_acc:.4f} at epoch {best_epoch}.",
            }
        )

    if "val_loss" in df.columns and df["val_loss"].notna().any():
        best_idx = int(df["val_loss"].idxmin())
        best_epoch = int(df.loc[best_idx, "epoch"])
        best_val_loss = float(df.loc[best_idx, "val_loss"])
        docs.append(
            {
                "title": f"Validation loss minimum: {run.name}",
                "text": f"{run.name} reaches its lowest validation loss of {best_val_loss:.4f} at epoch {best_epoch}.",
            }
        )

    if "train_accuracy" in df.columns and "val_accuracy" in df.columns:
        end_gap = float((df["train_accuracy"] - df["val_accuracy"]).tail(min(5, len(df))).mean())
        docs.append(
            {
                "title": f"Generalization gap: {run.name}",
                "text": f"{run.name} ends with an average train-validation accuracy gap of {end_gap:.4f} over the last few epochs.",
            }
        )

    if diag.overfitting_epoch is not None:
        docs.append(
            {
                "title": f"Overfitting signal: {run.name}",
                "text": f"TrainScope detects likely overfitting for {run.name} beginning around epoch {diag.overfitting_epoch}.",
            }
        )
    if diag.divergence_epoch is not None:
        docs.append(
            {
                "title": f"Instability signal: {run.name}",
                "text": f"TrainScope detects an instability spike for {run.name} around epoch {diag.divergence_epoch}.",
            }
        )
    return docs


def retrieve_evidence(question: str, docs: list[dict], top_k: int = 5) -> list[dict]:
    if not docs:
        return []
    corpus = [d["text"] for d in docs]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus + [question])
    sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    return [docs[i] for i in top_idx]


def generate_ai_answer(question: str, runs: List[RunData], comparison_df: pd.DataFrame, ranking_metric: str) -> AIResponse:
    docs = build_knowledge_base(runs, comparison_df, ranking_metric)
    evidence = retrieve_evidence(question, docs)
    evidence_text = "\n".join(f"- {item['title']}: {item['text']}" for item in evidence)

    generator = _load_generator()
    if generator is not None:
        prompt = (
            "You are TrainScope, an AI assistant for ML experiment diagnostics. "
            "Answer the question using only the evidence below. Be concise, specific, and grounded. "
            "If the evidence is weak, say so explicitly.\n\n"
            f"Question: {question}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            "Return 1 short paragraph and mention relevant run names."
        )
        try:
            result = generator(prompt)[0]["generated_text"].strip()
            return AIResponse(answer=result, evidence=[item["text"] for item in evidence], mode="local-llm")
        except Exception:
            pass

    return AIResponse(
        answer=_fallback_answer(question, evidence, comparison_df, ranking_metric),
        evidence=[item["text"] for item in evidence],
        mode="retrieval+rules",
    )


def _fallback_answer(question: str, evidence: list[dict], comparison_df: pd.DataFrame, ranking_metric: str) -> str:
    q = question.lower()
    best = comparison_df.iloc[0]
    if "why" in q and "best" in q:
        return (
            f"**{best['run_name']}** is currently top-ranked by {ranking_metric}. "
            f"It combines strong validation metrics with a stability score of {best['stability_score']:.2f}."
        )
    if "compare" in q or "difference" in q:
        if len(comparison_df) < 2:
            return "I need at least two runs to make a meaningful comparison."
        second = comparison_df.iloc[1]
        return (
            f"The strongest current run is **{best['run_name']}**, while **{second['run_name']}** is the next best. "
            f"The main differences come from validation behavior, stability, and whether TrainScope detected overfitting or instability."
        )
    joined = " ".join(item["text"] for item in evidence[:3])
    if joined:
        return joined
    return (
        "I could not find strong evidence for that question yet. Try asking about stability, overfitting, divergence, "
        "generalization, or why one run outperformed another."
    )
