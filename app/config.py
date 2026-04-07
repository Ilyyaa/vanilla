from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    db_host: str
    db_port: int
    db_user: str
    db_password: str
    db_name: str
    pg_table_name: str
    embed_dim: int
    text_search_config: str
    llm_model: str
    embed_model: str
    llm_request_timeout: float
    retriever_top_k: int
    retriever_sparse_top_k: int
    reranker_model: str
    reranker_top_n: int
    data_txt_dir: str
    txt_chunk_size: int
    txt_chunk_overlap: int


def _load_dotenv(env_file: str = ".env") -> None:
    path = Path(env_file)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_settings() -> Settings:
    _load_dotenv()

    return Settings(
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=int(os.getenv("DB_PORT", "5433")),
        db_user=os.getenv("DB_USER", "rag_user"),
        db_password=os.getenv("DB_PASSWORD", "password"),
        db_name=os.getenv("DB_NAME", "rag_db"),
        pg_table_name=os.getenv("PG_TABLE_NAME", "llamaindex_vectors"),
        embed_dim=int(os.getenv("EMBED_DIM", "1024")),
        text_search_config=os.getenv("TEXT_SEARCH_CONFIG", "russian"),
        llm_model=os.getenv("LLM_MODEL", "qwen2.5:1.5b"),
        embed_model=os.getenv("EMBED_MODEL", "bge-m3"),
        llm_request_timeout=float(os.getenv("LLM_REQUEST_TIMEOUT", "960.0")),
        retriever_top_k=int(os.getenv("RETRIEVER_TOP_K", "30")),
        retriever_sparse_top_k=int(os.getenv("RETRIEVER_SPARSE_TOP_K", "30")),
        reranker_model=os.getenv("RERANKER_MODEL", "models/cross-encoder-russian-msmarco"),
        reranker_top_n=int(os.getenv("RERANKER_TOP_N", "5")),
        data_txt_dir=os.getenv("DATA_TXT_DIR", "data_txt"),
        txt_chunk_size=int(os.getenv("TXT_CHUNK_SIZE", "700")),
        txt_chunk_overlap=int(os.getenv("TXT_CHUNK_OVERLAP", "100")),
    )
