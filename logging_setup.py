# logging_setup.py
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from config import LOG_DIR, LOG_LEVEL, LOG_NAME


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    fmt = "%(asctime)s [%(levelname).1s] %(name)s:%(lineno)d -> %(message)s"
    date = "%m-%d %H:%M:%S"

    handlers = [
        RotatingFileHandler(
            filename=f"{LOG_DIR}/run.log",
            maxBytes=5_000_000,
            backupCount=10,
            encoding="utf-8",
        ),
        logging.StreamHandler(sys.stdout),
    ]

    logging.basicConfig(
        level=LOG_LEVEL,
        format=fmt,
        datefmt=date,
        handlers=handlers,
        force=True,
    )
    return logging.getLogger(LOG_NAME)
