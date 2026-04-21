from typing import List

import pandas as pd

from analysis.diagnostics import RunDiagnostics, diagnose_run
from analysis.parser import RunData


def build_run_report(run_name: str, diag: RunDiagnostics) -> str:
    pieces = [f'**{run_name}** looks {diag.tags[0]}.']
    if diag.best_val_accuracy is not None:
        pieces.append(f'Best validation accuracy: {diag.best_val_accuracy:.4f}.')
    if diag.best_val_loss is not None:
        pieces.append(f'Best validation loss: {diag.best_val_loss:.4f}.')
    if diag.overfitting_epoch is not None:
        pieces.append(f'Overfitting likely begins around epoch {diag.overfitting_epoch}.')
    if diag.divergence_epoch is not None:
        pieces.append(f'An instability spike appears around epoch {diag.divergence_epoch}.')
    pieces.append(f'Stability score: {diag.stability_score:.2f}.')
    return ' '.join(pieces)


def build_comparison_report(comparison_df: pd.DataFrame, ranking_metric: str) -> str:
    best = comparison_df.iloc[0]
    worst = comparison_df.iloc[-1]
    if ranking_metric == 'val_accuracy':
        metric_text = f"the strongest validation accuracy is **{best['run_name']}**"
    elif ranking_metric == 'val_loss':
        metric_text = f"the lowest validation loss is achieved by **{best['run_name']}**"
    else:
        metric_text = f"the top-ranked run is **{best['run_name']}**"
    return (
        f"Based on **{ranking_metric}**, {metric_text}. "
        f"The least favorable run in the current ranking is **{worst['run_name']}**. "
        f"Use the diagnostic tags to understand whether the ranking is driven by accuracy, loss, or training stability."
    )


def answer_query(query: str, runs: List[RunData], comparison_df: pd.DataFrame, ranking_metric: str) -> str:
    q = query.lower()
    if 'stable' in q:
        best = comparison_df.sort_values('stability_score', ascending=False).iloc[0]
        return f"The most stable run is **{best['run_name']}** with a stability score of {best['stability_score']:.2f}."
    if 'overfit' in q or 'overfitting' in q:
        overfit_rows = comparison_df[comparison_df['overfitting_epoch'].notna()]
        if overfit_rows.empty:
            return 'I do not see a strong overfitting signal in the current runs.'
        formatted = ', '.join(
            f"**{row.run_name}** around epoch {int(row.overfitting_epoch)}" for row in overfit_rows.itertuples()
        )
        return f'TrainScope detected overfitting in: {formatted}.'
    if 'best' in q or 'top' in q:
        best = comparison_df.iloc[0]
        return f"The best run by **{ranking_metric}** is **{best['run_name']}**."
    if 'generalize' in q or 'generalization' in q:
        candidates = []
        for run in runs:
            diag = diagnose_run(run.data)
            candidates.append((run.name, diag.stability_score, diag.best_val_accuracy or 0.0, len(diag.warnings)))
        candidates.sort(key=lambda x: (x[2], x[1], -x[3]), reverse=True)
        winner = candidates[0][0]
        return f'Based on validation behavior and stability, **{winner}** appears to generalize best among the current runs.'
    return (
        'I can answer questions about stability, overfitting, divergence, and best-performing runs. '
        'Try asking: “Which run is most stable?” or “Which run overfit?”'
    )
