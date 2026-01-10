from __future__ import annotations
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml

def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    base = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base
