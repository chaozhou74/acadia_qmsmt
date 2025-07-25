import logging
import re
from contextlib import contextmanager
from functools import wraps


@contextmanager
def suppress_log_messages(patterns, logger_names=[""], levels=(logging.WARNING, logging.ERROR)):
    """
    Suppress log records matching any regex in `patterns`, for given `logger_names` and levels.
    """
    class SuppressFilter(logging.Filter):
        def filter(self, record):
            return not (
                record.levelno in levels and any(re.search(p, record.getMessage()) for p in patterns)
            )

    loggers = [logging.getLogger(name) for name in logger_names]
    filters = []

    try:
        for logger in loggers:
            for handler in logger.handlers:
                f = SuppressFilter()
                handler.addFilter(f)
                filters.append((handler, f))
        yield
    finally:
        for handler, f in filters:
            handler.removeFilter(f)


@contextmanager
def suppress_data_sync_messages(enabled=True):
    """
    Suppress common DataManager sync warnings and errors when `enabled=True`.
    """
    if not enabled:
        yield
        return

    patterns = [
        r"Unable to connect to target DataManager",
        r"Exception synchronizing",
        r"Socket peer closed connection"
    ]
    logger_names = ["", "acadia", "acadia_qmsmt"]
    levels = (logging.WARNING, logging.ERROR)

    with suppress_log_messages(patterns, logger_names, levels):
        yield