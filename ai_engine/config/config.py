"""
config.py  —  Singleton configuration loader.

Usage anywhere in the project:
    from config.config import cfg
    threshold = cfg.seat_zones.max_vacancy_dist
"""
from __future__ import annotations

import os
import yaml
from pathlib import Path


class _DotDict(dict):
    """dict subclass with attribute-style access.  cfg.section.key works."""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def from_dict(cls, d: dict) -> "_DotDict":
        obj = cls()
        for k, v in d.items():
            obj[k] = cls.from_dict(v) if isinstance(v, dict) else v
        return obj


class _Config:
    _instance: "_Config | None" = None
    _data:     "_DotDict | None" = None

    def __new__(cls) -> "_Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load(self, path: str | Path | None = None) -> None:
        if path is None:
            path = Path(__file__).parent / "config.yaml"
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self._data = _DotDict.from_dict(raw)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if self._data is None:
            self._load()
        return self._data[name]

    def reload(self, path: str | Path | None = None) -> None:
        """Force reload — useful for testing with a different config."""
        self._load(path)

    # ── Typed convenience accessors ──────────────────────────────────────────

    @property
    def model_dir(self) -> Path:
        return Path(self.paths.model_dir)

    @property
    def log_dir(self) -> Path:
        p = Path(self.paths.log_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def evidence_dir(self) -> Path:
        p = Path(self.paths.evidence_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def output_dir(self) -> Path:
        p = Path(self.paths.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def ai_frame_size(self) -> tuple[int, int]:
        s = self.inference.ai_frame_size
        return (int(s[0]), int(s[1]))


# Public singleton — import this everywhere
cfg = _Config()
