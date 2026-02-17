import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE_PATH = PROJECT_ROOT / ".env"
_ENV_LOADED = False


def _parse_env_line(line: str) -> Optional[tuple[str, str]]:
    """
    Parse one .env line in `KEY=VALUE` format.

    Supports:
    - leading/trailing spaces
    - `export KEY=VALUE`
    - quoted values: "..." or '...'
    - comments via full-line `#`
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    if stripped.startswith("export "):
        stripped = stripped[len("export ") :].strip()

    if "=" not in stripped:
        logger.warning("Skip invalid .env line (missing '='): %s", line.rstrip("\n"))
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()

    if not key:
        logger.warning("Skip invalid .env line (empty key): %s", line.rstrip("\n"))
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]

    return key, value


def load_env_file(override: bool = False) -> None:
    """
    Load environment variables from project root `.env` if it exists.

    Args:
        override: If True, overwrite existing process environment variables.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return

    if not ENV_FILE_PATH.exists():
        logger.info("No .env file found at %s", ENV_FILE_PATH)
        _ENV_LOADED = True
        return

    loaded_count = 0
    with ENV_FILE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            parsed = _parse_env_line(line)
            if not parsed:
                continue
            key, value = parsed
            if key in os.environ and not override:
                continue
            os.environ[key] = value
            loaded_count += 1

    _ENV_LOADED = True
    logger.info("Loaded %d env vars from %s", loaded_count, ENV_FILE_PATH)


def get_required_env(key: str) -> str:
    """
    Read a required environment variable after loading project .env.

    Args:
        key: Environment variable name.

    Returns:
        Environment variable value.

    Raises:
        ValueError: If the variable is missing or empty.
    """
    load_env_file()
    value = os.getenv(key, "").strip()
    if not value:
        logger.error("Missing required environment variable: %s", key)
        raise ValueError(f"{key} is required but not found.")
    return value
