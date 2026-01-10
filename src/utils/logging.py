from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd

@dataclass
class RunLogger:
    run_dir: Path

    def __post_init__(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_json(self, name: str, obj: Dict[str, Any]) -> None:
        path = self.run_dir / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

    def save_df(self, name: str, df: pd.DataFrame) -> None:
        path = self.run_dir / name
        df.to_csv(path, index=False)

def append_summary(summary_csv: Path, row: Dict[str, Any]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(summary_csv, index=False)
