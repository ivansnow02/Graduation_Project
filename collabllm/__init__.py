"""
collabllm
~~~~~~~~~
Package initialisation: version metadata, global configuration flags,
and one-time setup (logging, runtime directories, …).
"""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path


# Avoid importing setuptools at package import time; provide a tiny local
# strtobool equivalent to parse environment boolean-like strings. This
# avoids adding a hard runtime dependency on setuptools and is robust.
def _local_strtobool(val: str) -> int:
    """Return 1 for truthy strings, 0 for falsy strings, else raise ValueError.

    Mirrors distutils.util.strtobool / setuptools.util.strtobool behavior
    sufficiently for our usage.
    """
    v = str(val).strip().lower()
    if v in ("1", "y", "yes", "true", "on"):
        return 1
    if v in ("0", "n", "no", "false", "off"):
        return 0
    raise ValueError(f"invalid truth value {val!r}")


# --------------------------------------------------------------------------- #
# Public package metadata                                                     #
# --------------------------------------------------------------------------- #
__version__ = "0.1.0"  # update as needed
__author__ = "Shirley Wu & the CollabLLM team"

__all__ = [
    "__version__",
    "ENABLE_COLLABLLM_LOGGING",
    "RUN_USER_DIR",
]


# --------------------------------------------------------------------------- #
# Utility: boolean env-var parser                                             #
# --------------------------------------------------------------------------- #
def _env_flag(name: str, default: str = "1") -> bool:
    """
    Convert an environment variable to bool.

    Truthy strings : "1", "true", "yes", "on"   (case-insensitive)
    Falsy  strings : "0", "false", "no", "off"
    """
    try:
        return bool(_local_strtobool(os.getenv(name, default)))
    except ValueError:
        # Invalid value; fall back to default.
        return bool(_local_strtobool(default))


# --------------------------------------------------------------------------- #
# Global logging switch                                                       #
# --------------------------------------------------------------------------- #
ENABLE_COLLABLLM_LOGGING: bool = _env_flag("ENABLE_COLLABLLM_LOGGING", "1")


_pkg_logger = logging.getLogger("collabllm")

if ENABLE_COLLABLLM_LOGGING:
    # Configure basic console output if the user hasn’t configured logging yet.
    # We guard with "if not root.handlers" to avoid double-configuration.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    _pkg_logger.info("CollabLLM logging enabled.")
else:
    # Silence *all* log records emitted from collabllm.* by:
    # 1) setting a level higher than CRITICAL
    # 2) preventing propagation to the root logger
    # 3) attaching a NullHandler
    _pkg_logger.setLevel(logging.CRITICAL)
    _pkg_logger.propagate = False
    _pkg_logger.handlers.clear()
    _pkg_logger.addHandler(logging.NullHandler())


# --------------------------------------------------------------------------- #
# LiteLLM                                                                     #
# --------------------------------------------------------------------------- #
import litellm

# Diable the cache by default, as it is not needed in most cases.
litellm.disable_cache()

# Also silence the LiteLLM logger.
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
_pkg_logger.info("Disable LiteLLM cache and logging by default. ")

# --------------------------------------------------------------------------- #
# Per-user runtime directory                                                  #
# --------------------------------------------------------------------------- #
_DEFAULT_RUN_DIR = "run/collabllm/user_{uid}"


def _resolve_run_user_dir() -> Path:
    # 1) honour explicit env-var
    env_val = os.getenv("RUN_USER_DIR")
    if env_val:
        return Path(env_val).expanduser()

    # 2) fall back to XDG-runtime-style path
    return Path(_DEFAULT_RUN_DIR.format(uid=os.getuid()))


RUN_USER_DIR: Path = _resolve_run_user_dir()
os.environ["RUN_USER_DIR"] = str(RUN_USER_DIR)

try:
    RUN_USER_DIR.mkdir(parents=True, exist_ok=True)
except OSError as exc:
    # If the runtime mount is not writable (EROFS) or missing (ENOENT)
    # fall back to a per-user cache directory. This handles macOS where
    # `/run` may not exist or is read-only.
    if exc.errno in {errno.EACCES, errno.ENOENT, errno.EROFS}:
        fallback = Path.home() / ".cache" / "collabllm"
        fallback.mkdir(parents=True, exist_ok=True)
        _pkg_logger.warning(
            "Cannot access %s; using %s instead.", RUN_USER_DIR, fallback
        )
        RUN_USER_DIR = fallback
    else:
        raise
