from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import pandas as pd


@dataclass
class RunData:
    name: str
    data: pd.DataFrame


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if 'epoch' not in df.columns:
        df.insert(0, 'epoch', range(1, len(df) + 1))
    return df


def parse_single_run(name: str, data: pd.DataFrame) -> RunData:
    df = _normalize_columns(data)
    numeric_cols = [c for c in df.columns if c != 'epoch']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(how='all', subset=numeric_cols).reset_index(drop=True)
    return RunData(name=name, data=df)


def load_runs_from_uploads(uploaded_files: Iterable) -> List[RunData]:
    runs = []
    for file in uploaded_files:
        df = pd.read_csv(file)
        name = Path(file.name).stem
        runs.append(parse_single_run(name, df))
    return runs


def load_sample_runs(sample_dir: Path) -> List[RunData]:
    runs = []
    for path in sorted(sample_dir.glob('*.csv')):
        df = pd.read_csv(path)
        runs.append(parse_single_run(path.stem, df))
    return runs
