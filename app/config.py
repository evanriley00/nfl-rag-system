import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_dotenv_file(Path(".env"))


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str
    data_dir: Path
    docs_dir: Path
    index_dir: Path
    top_k: int
    host: str
    port: int
    openai_api_key: str
    embedding_model: str
    embedding_dimensions: int
    generation_model: str
    chroma_collection_name: str


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


settings = Settings(
    app_name=os.getenv("GRIDMIND_APP_NAME", "GridMind NFL RAG"),
    data_dir=Path(os.getenv("GRIDMIND_DATA_DIR", "./data")),
    docs_dir=Path(os.getenv("GRIDMIND_DOCS_DIR", "./data/documents")),
    index_dir=Path(os.getenv("GRIDMIND_INDEX_DIR", "./data/index")),
    top_k=_int_env("GRIDMIND_TOP_K", 4),
    host=os.getenv("GRIDMIND_HOST", "127.0.0.1"),
    port=_int_env("GRIDMIND_PORT", 8000),
    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    embedding_model=os.getenv("GRIDMIND_EMBEDDING_MODEL", "text-embedding-3-small"),
    embedding_dimensions=_int_env("GRIDMIND_EMBEDDING_DIMENSIONS", 512),
    generation_model=os.getenv("GRIDMIND_GENERATION_MODEL", "gpt-5-mini"),
    chroma_collection_name=os.getenv("GRIDMIND_CHROMA_COLLECTION", "gridmind_chunks"),
)
