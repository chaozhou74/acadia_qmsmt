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


def add_data_sync_log_filter(grace_period=5.0):
    """Hide the harmless DataManager sync warnings during start-up only.

    Right after a run starts, acadia keeps trying to reach the data-sync target
    and logs "Unable to connect to target DataManager" (plus a couple of related
    messages) until the connection comes up. That early chatter is normal, so
    this hides it for the first `grace_period` seconds. If the connection still
    isn't up after that, the warnings come through again — so a real, persistent
    connection problem is not hidden from you.

    Call once per run (deploy() does this for you). Calling it again just
    restarts the grace window; it won't stack duplicate filters.
    """
    import time

    patterns = [
        "Unable to connect to target DataManager",
        "Exception synchronizing",
        "Socket peer closed connection",
    ]

    class _Filter(logging.Filter):
        _data_sync = True

        def __init__(self):
            super().__init__()
            self.start = time.monotonic()

        def filter(self, record):
            if any(p in record.getMessage() for p in patterns):
                # inside the grace window -> drop; after it -> let through
                return time.monotonic() - self.start >= grace_period
            return True

    for name in ("acadia", "acadia_qmsmt"):
        lg = logging.getLogger(name)
        existing = next((f for f in lg.filters if getattr(f, "_data_sync", False)), None)
        if existing is not None:
            existing.start = time.monotonic()  # restart the grace window for this run
        else:
            lg.addFilter(_Filter())