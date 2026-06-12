from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    student_id: str
    student_port: int
    teacher_base_url: str
    llm_model: str
    llm_timeout_seconds: float
    embedding_model_path: Path
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_context_chars: int
    log_path: Path

    @property
    def teacher_proxy_base_url(self) -> str:
        return f"{self.teacher_base_url}/proxy"


def get_settings() -> Settings:
    chunk_size = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "180"))
    if chunk_overlap >= chunk_size:
        raise RuntimeError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    return Settings(
        student_id=os.getenv("STUDENT_ID", "B22DCVT351"),
        student_port=int(os.getenv("STUDENT_PORT", "5000")),
        teacher_base_url=os.getenv(
            "TEACHER_BASE_URL",
            "http://192.168.50.218:8000/api/v1",
        ).rstrip("/"),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "45")),
        embedding_model_path=_resolve_project_path(
            os.getenv("EMBEDDING_MODEL_PATH", "models/vietnamese-sbert")
        ),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=int(os.getenv("TOP_K", "8")),
        max_context_chars=int(os.getenv("MAX_CONTEXT_CHARS", "9000")),
        log_path=_resolve_project_path(os.getenv("LOG_PATH", "logs/ask_logs.jsonl")),
    )
