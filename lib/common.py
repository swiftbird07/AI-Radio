from __future__ import annotations

import json
import os
import re
import secrets
import string
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DOTENV_LOADED = False


def load_project_env() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    if load_dotenv is None:
        raise RuntimeError(
            "python-dotenv is required. Install with: pip install python-dotenv"
        )
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)
    _DOTENV_LOADED = True


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(v) for v in value]
    if isinstance(value, str):
        return ENV_PATTERN.sub(lambda m: os.getenv(m.group(1), ""), value)
    return value


def load_config(config_path: str) -> dict[str, Any]:
    load_project_env()
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return expand_env_vars(raw)


def ensure_dir(path: str | Path) -> Path:
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def get_provider_config(config: dict[str, Any], provider_type: str) -> dict[str, Any]:
    provider_type_upper = provider_type.upper()
    for provider in config.get("providers", []):
        if str(provider.get("type", "")).upper() == provider_type_upper:
            return provider.get("config", {})
    raise ValueError(f"Provider not found for type '{provider_type}'")


def resolve_workdir(path: str | None) -> Path:
    return ensure_dir(path or ".tmp")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "section"


def _new_run_id(length: int = 4) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def get_or_create_run_id(workdir: str | Path, force_new: bool = False) -> str:
    wd = ensure_dir(workdir)
    run_id_file = wd / "run_id.txt"
    if not force_new and run_id_file.exists():
        current = run_id_file.read_text(encoding="utf-8").strip()
        if re.fullmatch(r"[A-Za-z0-9]{4}", current or ""):
            return current
    run_id = _new_run_id()
    run_id_file.write_text(run_id + "\n", encoding="utf-8")
    return run_id
