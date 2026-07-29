"""Machine- and board-specific paths for the validation harness.

These are read from a gitignored ``paths.local.yaml`` next to this file (copy
``paths.local.example.yaml`` to create it) so that no lab IP or data path is
committed to the package. Environment variables override the file:

    SEQVIZ_BOARD_IP            -> board_ip
    SEQVIZ_LOOPBACK_DATA_ROOT  -> loopback_data_root

The pure off-hardware paths -- ``timing_validation.py --dryrun`` and the core
module itself -- need none of this; only deploying (``--case``/``--all``),
re-analysing archived runs (``--revalidate``/``--analyse``) and ``selftest.py``
do, and each asks for the specific key it needs via :func:`require`.
"""
import os
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_LOCAL = _HERE / "paths.local.yaml"

_ENV = {
    "board_ip": "SEQVIZ_BOARD_IP",
    "loopback_data_root": "SEQVIZ_LOOPBACK_DATA_ROOT",
}


def load():
    """The local config as a dict (empty if no file and no env vars)."""
    data = {}
    if _LOCAL.is_file():
        data = yaml.safe_load(_LOCAL.read_text()) or {}
    for key, env in _ENV.items():
        if os.environ.get(env):
            data[key] = os.environ[env]
    return data


def require(key):
    """Return ``load()[key]`` or exit with a message pointing at the example file."""
    value = load().get(key)
    if not value:
        raise SystemExit(
            f"validation needs '{key}', which is not set. Copy "
            f"validation/paths.local.example.yaml to validation/paths.local.yaml "
            f"and fill it in (or set the {_ENV.get(key, 'matching')} env var).")
    return value
