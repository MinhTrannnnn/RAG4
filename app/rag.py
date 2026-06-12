from __future__ import annotations

import os
import pickle
import re
from dataclasses import dataclass
from threading import Lock

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import Settings


VECTOR_DB_SCHEMA_VERSION = 1
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
SPLIT_PATTERN = re.compile(
    r"\n\s*\n+|\n(?=\s*(?:Chuong|Bai|Muc|Cau|Slide|Phan|Dieu)\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    embedding: np.ndarray
    tokens: set[str]


@dataclass(frozen=True)
class ScoredChunk:
    score: float
    chunk: Chunk


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_PATTERN.findall(text) if len(token) > 1}


def split_long_piece(piece: str, chunk_size: int, overlap: int) -> list[str]:
    piece = re.sub(r"\s+", " ", piece).strip()
    if len(piece) <= chunk_size:
        return [piece] if piece else []

    chunks: list[str] = []
    start = 0
    while start < len(piece):
        end = min(start + chunk_size, len(piece))
        if end < len(piece):
            boundary = max(
                piece.rfind(". ", start, end),
                piece.rfind("? ", start, end),
                piece.rfind("! ", start, end),
                piece.rfind("; ", start, end),
            )
            if boundary > start + chunk_size // 2:
                end = boundary + 1

        text = piece[start:end].strip()
        if text:
            chunks.append(text)

        if end >= len(piece):
            break
        start = max(end - overlap, start + 1)

    return chunks


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    result: list[str] = []
    current = ""
    pieces = [
        re.sub(r"\s+", " ", piece).strip()
        for piece in SPLIT_PATTERN.split(normalized)
        if piece.strip()
    ]

    for piece in pieces:
        if len(piece) > chunk_size:
            if current:
                result.append(current)
                current = ""
            result.extend(split_long_piece(piece, chunk_size, overlap))
            continue

        candidate = f"{current}\n\n{piece}".strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                result.append(current)
            current = piece

    if current:
        result.append(current)

    return result


def build_context(
    scored_chunks: list[ScoredChunk],
    max_context_chars: int,
) -> tuple[str, list[Chunk]]:
    selected: list[Chunk] = []
    context_parts: list[str] = []
    total_chars = 0

    for scored in scored_chunks:
        part = f"[{scored.chunk.chunk_id} | score={scored.score:.3f}]\n{scored.chunk.text}"
        if context_parts and total_chars + len(part) > max_context_chars:
            break
        selected.append(scored.chunk)
        context_parts.append(part)
        total_chars += len(part)

    return "\n\n".join(context_parts), selected


class RagService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: SentenceTransformer | None = None
        self._document_id: str | None = None
        self._chunks: list[Chunk] = []
        self._lock = Lock()
        self._load_persisted_index()

    @property
    def chunk_count(self) -> int:
        with self._lock:
            return len(self._chunks)

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            if not self._settings.embedding_model_path.is_dir():
                raise RuntimeError(
                    f"Embedding model path not found: {self._settings.embedding_model_path}. "
                    "Run: .\\.venv\\Scripts\\python.exe scripts/download_embedding_model.py "
                    "while online, then start the server again."
                )
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            self._model = SentenceTransformer(
                str(self._settings.embedding_model_path),
                local_files_only=True,
            )
        return self._model

    def _load_persisted_index(self) -> None:
        path = self._settings.vector_db_path
        if not path.is_file():
            return

        with path.open("rb") as file:
            payload = pickle.load(file)

        if payload.get("version") != VECTOR_DB_SCHEMA_VERSION:
            raise RuntimeError(f"Unsupported vector DB version in {path}")

        chunks = payload.get("chunks")
        if not isinstance(chunks, list):
            raise RuntimeError(f"Invalid vector DB file: {path}")

        with self._lock:
            self._document_id = payload.get("document_id")
            self._chunks = chunks

    def _save_persisted_index(self, doc_id: str | None, chunks: list[Chunk]) -> None:
        path = self._settings.vector_db_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        payload = {
            "version": VECTOR_DB_SCHEMA_VERSION,
            "document_id": doc_id,
            "chunks": chunks,
        }

        with tmp_path.open("wb") as file:
            pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)

    def ingest(self, text: str, doc_id: str | None) -> tuple[str | None, int]:
        chunk_texts = split_text(
            text,
            chunk_size=self._settings.chunk_size,
            overlap=self._settings.chunk_overlap,
        )
        if not chunk_texts:
            raise ValueError("Document did not produce any chunks")

        model = self._get_model()
        embeddings = model.encode(
            chunk_texts,
            batch_size=16,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        chunks = [
            Chunk(
                chunk_id=f"chunk_{index + 1}",
                text=chunk_text,
                embedding=embeddings[index],
                tokens=tokenize(chunk_text),
            )
            for index, chunk_text in enumerate(chunk_texts)
        ]

        with self._lock:
            self._document_id = doc_id
            self._chunks = chunks

        self._save_persisted_index(doc_id, chunks)
        return doc_id, len(chunks)

    def retrieve(self, question: str, top_k: int | None = None) -> list[ScoredChunk]:
        with self._lock:
            chunks = list(self._chunks)

        if not chunks:
            raise LookupError("No document has been uploaded yet")

        model = self._get_model()
        query_embeddings = model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        query_embedding = query_embeddings[0]
        lexical_tokens = tokenize(question)

        scored_chunks: list[ScoredChunk] = []
        for chunk in chunks:
            semantic_score = float(np.dot(query_embedding, chunk.embedding))
            overlap = lexical_tokens & chunk.tokens
            lexical_score = len(overlap) / max(len(lexical_tokens), 1)
            score = semantic_score + (0.08 * lexical_score)
            scored_chunks.append(ScoredChunk(score=score, chunk=chunk))

        scored_chunks.sort(key=lambda item: item.score, reverse=True)
        return scored_chunks[: top_k or self._settings.top_k]
