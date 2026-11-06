"""Boolean environment-variable switches (stdlib only, safe to import early)."""

import os

_OFF = {"0", "false", "no", "off"}


def env_enabled(name, default=True):
    """False if ``name`` is 0/false/no/off (case-insensitive, stripped).

    Unset or empty -> ``default``; any other value -> True.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in _OFF
