"""Small YAML helpers and the class list."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import yaml

from mv.paths import CONFIGS


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def classes() -> dict:
    return load_yaml(CONFIGS / "classes.yaml")


def class_names() -> list[str]:
    names = classes()["names"]
    return [names[i] for i in sorted(names)]


def class_id(name: str) -> int:
    return class_names().index(name)


def fraktion(name: str) -> str:
    return classes()["fraktion"][name]


def config_hash(cfg: dict) -> str:
    """Short stable hash of a config dict, logged with every run."""
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
