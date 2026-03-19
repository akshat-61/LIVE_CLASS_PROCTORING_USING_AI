"""
logging_setup.py  —  Central logging configuration.

Call once at process startup:
    from config.logging_setup import setup_logging
    setup_logging()

Then anywhere:
    import logging
    log = logging.getLogger(__name__)
    log.info("Seat zones locked: %d seats", n)
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path


_LEVEL_MAP = {
    "DEBUG":   logging.DEBUG,
    "INFO":    logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR":   logging.ERROR,
}

_FMT      = "%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s"
_FMT_DATE = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level:        str  = "INFO",
    log_dir:      str  = "logs",
    console:      bool = True,
    to_file:      bool = True,
    max_bytes:    int  = 10 * 1024 * 1024,   # 10 MB
    backup_count: int  = 5,
) -> None:
    """
    Configure the root logger with optional console and rotating-file handlers.
    Safe to call multiple times — subsequent calls are no-ops unless force=True.
    """
    root = logging.getLogger()

    # Already configured — skip
    if root.handlers:
        return

    numeric_level = _LEVEL_MAP.get(level.upper(), logging.INFO)
    root.setLevel(numeric_level)

    formatter = logging.Formatter(_FMT, datefmt=_FMT_DATE)

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(numeric_level)
        ch.setFormatter(formatter)
        root.addHandler(ch)

    if to_file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir) / "proctoring.log"
        fh = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        fh.setLevel(numeric_level)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Suppress noisy third-party loggers
    for noisy in ("ultralytics", "urllib3", "PIL", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).debug("Logging initialised — level=%s", level)


def get_logger(name: str) -> logging.Logger:
    """Shorthand: from config.logging_setup import get_logger"""
    return logging.getLogger(name)


def setup_from_cfg() -> None:
    """Bootstrap logging directly from config.yaml."""
    try:
        from config.config import cfg
        lc = cfg.logging
        setup_logging(
            level        = str(lc.level),
            log_dir      = str(cfg.paths.log_dir),
            console      = bool(lc.console),
            to_file      = bool(lc.file),
            max_bytes    = int(lc.max_bytes),
            backup_count = int(lc.backup_count),
        )
    except Exception as e:
        # Fallback: at least get console logging working
        setup_logging(level="INFO")
        logging.getLogger(__name__).warning(
            "Could not load config for logging setup: %s — using defaults", e
        )
