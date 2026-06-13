from __future__ import annotations

# Load the encoding/safety bootstrap as early as possible so every psai
# process (pytest, the sidecar build, and the packaged runtime) gets the
# same surrogate-safe JSON and SQLite behavior.
from . import _encoding_safety as _encoding_safety  # noqa: F401

__all__ = ["__version__"]
__version__ = "0.1.0"
