from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from analysis.parser import RunData


@dataclass
class RunDiagnostics:
    best_val_accuracy: float | None
    best_val_loss: float | None
    stability_score: float
    tags: List[str]
    warnings: List[str]
    overfitting_epoch: int | None
    divergence_epoch: int | None


def _safe_std_ratio(values: pd.Series) -> float:
    values = values.dropna()
    if len(values) < 2:
        return 0.0
    mean_abs = max(abs(values.mean()), 1e-8)
    return float(values.std() / mean_abs)


def diagnose_run(df: pd.DataFrame) -> RunDiagnostics:
    tags: List[str] = []
    warnings: List[str] = []
    overfitting_epoch = None
    divergence_epoch = None

    best_val_accuracy = float(df['val_accuracy'].max()) if 'val_accuracy' in df.columns and df['val_accuracy'].notna().any() else None
    best_val_loss = float(df['val_loss'].min()) if 'val_loss' in df.columns and df['val_loss'].notna().any() else None

    train_loss_var = _safe_std_ratio(df['train_loss']) if 'train_loss' in df.columns else 0.0
    val_loss_var = _safe_std_ratio(df['val_loss']) if 'val_loss' in df.columns else 0.0
    gap = 0.0
    if 'train_accuracy' in df.columns and 'val_accuracy' in df.columns:
        gap = float((df['train_accuracy'] - df['val_accuracy']).tail(min(5, len(df))).mean())

    stability_score = max(0.0, 1.0 - min(1.0, 0.5 * train_loss_var + 0.5 * val_loss_var))

    if 'val_loss' in df.columns and 'train_loss' in df.columns and len(df) >= 6:
        best_idx = int(df['val_loss'].idxmin())
        tail = df.iloc[best_idx + 1:]
        if len(tail) >= 3 and tail['val_loss'].mean() > df.loc[best_idx, 'val_loss'] * 1.05 and tail['train_loss'].mean() <= df.loc[best_idx, 'train_loss'] * 1.02:
            overfitting_epoch = int(df.loc[best_idx, 'epoch'])
            tags.append('overfitting detected')
            warnings.append(f'Validation loss worsened after epoch {overfitting_epoch} while training loss kept improving.')

    if 'val_loss' in df.columns:
        spikes = df['val_loss'].pct_change().replace([np.inf, -np.inf], np.nan)
        spike_rows = df.loc[spikes > 0.35]
        if not spike_rows.empty:
            divergence_epoch = int(spike_rows.iloc[0]['epoch'])
            tags.append('instability spike')
            warnings.append(f'Validation loss spiked sharply around epoch {divergence_epoch}.')

    if stability_score > 0.82:
        tags.append('stable training')
    elif stability_score > 0.62:
        tags.append('moderately stable')
    else:
        tags.append('unstable training')
        warnings.append('Metric curves show high relative variance across epochs.')

    if gap > 0.08:
        tags.append('generalization gap')
        warnings.append('Training accuracy is noticeably ahead of validation accuracy near the end of training.')

    if 'val_accuracy' in df.columns and len(df) >= 5:
        recent = df['val_accuracy'].tail(5)
        if recent.max() - recent.min() < 0.005:
            tags.append('plateaued')

    if not tags:
        tags.append('insufficient signal')

    return RunDiagnostics(
        best_val_accuracy=best_val_accuracy,
        best_val_loss=best_val_loss,
        stability_score=stability_score,
        tags=tags,
        warnings=warnings,
        overfitting_epoch=overfitting_epoch,
        divergence_epoch=divergence_epoch,
    )


def compare_runs(runs: list[RunData], ranking_metric: str) -> pd.DataFrame:
    rows = []
    for run in runs:
        diag = diagnose_run(run.data)
        row = {
            'run_name': run.name,
            'stability_score': round(diag.stability_score, 4),
            'best_val_accuracy': round(diag.best_val_accuracy, 4) if diag.best_val_accuracy is not None else None,
            'best_val_loss': round(diag.best_val_loss, 4) if diag.best_val_loss is not None else None,
            'tags': ', '.join(diag.tags),
            'overfitting_epoch': diag.overfitting_epoch,
            'divergence_epoch': diag.divergence_epoch,
        }
        if ranking_metric == 'val_accuracy':
            score = diag.best_val_accuracy if diag.best_val_accuracy is not None else -np.inf
        elif ranking_metric == 'val_loss':
            score = -(diag.best_val_loss if diag.best_val_loss is not None else np.inf)
        elif ranking_metric == 'train_loss':
            score = -(float(run.data['train_loss'].min()) if 'train_loss' in run.data.columns else np.inf)
        else:
            score = diag.stability_score
        row['_ranking_score'] = score
        rows.append(row)

    out = pd.DataFrame(rows).sort_values('_ranking_score', ascending=False).drop(columns=['_ranking_score']).reset_index(drop=True)
    return out


def supported_columns_message() -> str:
    return (
        'TrainScope works best when each CSV contains epoch-wise metrics such as '\
        '`epoch`, `train_loss`, `val_loss`, `train_accuracy`, and `val_accuracy`. '\
        'Additional numeric columns can be added and inspected later.'
    )
