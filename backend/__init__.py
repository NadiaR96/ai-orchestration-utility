import os
from pathlib import Path


def _strip_wrapping_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_dotenv_if_present() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    dotenv_path = repo_root / ".env"
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue

        value = _strip_wrapping_quotes(value.strip())
        os.environ.setdefault(key, value)


def _ensure_hf_token_aliases() -> None:
    if os.getenv("HF_TOKEN"):
        return

    for alias in ("HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.getenv(alias)
        if token:
            os.environ["HF_TOKEN"] = token
            return


_load_dotenv_if_present()
_ensure_hf_token_aliases()