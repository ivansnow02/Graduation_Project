"""
collabllm
~~~~~~~~~
包初始化：版本元数据、全局配置开关，以及一次性设置（日志、运行目录等）。
"""

from __future__ import annotations

import errno
import logging
import os
from pathlib import Path


# 避免在包导入时依赖 setuptools；提供一个轻量的 strtobool 等价实现
# 用于解析环境变量中的布尔值字符串。这样可以避免将 setuptools
# 作为运行时强依赖，代码更稳健。
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
# 公共包元数据                                                                 #
# --------------------------------------------------------------------------- #
__version__ = "0.1.0"  # update as needed
__author__ = "Shirley Wu & the CollabLLM team"

__all__ = [
    "__version__",
    "ENABLE_COLLABLLM_LOGGING",
    "RUN_USER_DIR",
]


# --------------------------------------------------------------------------- #
# 工具：从环境变量解析布尔标志                                               #
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
        # 无效的值，回退到默认值。
        return bool(_local_strtobool(default))


# --------------------------------------------------------------------------- #
# 全局日志开关                                                                 #
# --------------------------------------------------------------------------- #
ENABLE_COLLABLLM_LOGGING: bool = _env_flag("ENABLE_COLLABLLM_LOGGING", "1")


_pkg_logger = logging.getLogger("collabllm")

if ENABLE_COLLABLLM_LOGGING:
    # 如果用户没有自定义日志配置，则设置基本的控制台输出格式。
    # 使用 "if not root.handlers" 以避免重复配置。
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    _pkg_logger.info("CollabLLM logging enabled.")
else:
    # 使 collabllm.* 的所有日志静默：
    # 1) 设置日志等级到 CRITICAL 以上
    # 2) 禁止向 root logger 传播
    # 3) 附加 NullHandler
    _pkg_logger.setLevel(logging.CRITICAL)
    _pkg_logger.propagate = False
    _pkg_logger.handlers.clear()
    _pkg_logger.addHandler(logging.NullHandler())


# --------------------------------------------------------------------------- #
# LiteLLM 相关设置                                                              #
# --------------------------------------------------------------------------- #
import litellm

# 默认禁用 LiteLLM 的缓存（大多数场景不需要）
litellm.disable_cache()

# 同时静默 LiteLLM 的日志
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
_pkg_logger.info("已默认禁用 LiteLLM 缓存与日志。")

# --------------------------------------------------------------------------- #
# 每用户运行时目录                                                              #
# --------------------------------------------------------------------------- #
_DEFAULT_RUN_DIR = "run/collabllm/user_{uid}"


def _resolve_run_user_dir() -> Path:
    # 1) 优先使用显式环境变量
    env_val = os.getenv("RUN_USER_DIR")
    if env_val:
        return Path(env_val).expanduser()

    # 2) 回退到类似 XDG runtime 的路径
    return Path(_DEFAULT_RUN_DIR.format(uid=os.getuid()))


RUN_USER_DIR: Path = _resolve_run_user_dir()
os.environ["RUN_USER_DIR"] = str(RUN_USER_DIR)

try:
    RUN_USER_DIR.mkdir(parents=True, exist_ok=True)
except OSError as exc:
    # 如果运行时目录不可写（如只读文件系统 EROFS）或不存在（ENOENT），
    # 则回退到每用户的 cache 目录。这解决了 macOS 上 `/run` 不存在或只读的问题。
    if exc.errno in {errno.EACCES, errno.ENOENT, errno.EROFS}:
        fallback = Path.home() / ".cache" / "collabllm"
        fallback.mkdir(parents=True, exist_ok=True)
        _pkg_logger.warning(
            "Cannot access %s; using %s instead.", RUN_USER_DIR, fallback
        )
        RUN_USER_DIR = fallback
    else:
        raise
