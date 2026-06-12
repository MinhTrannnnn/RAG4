from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import Settings, get_settings
from app.llm import (
    LlmAnswerParseError,
    LlmService,
    TeacherProxyRequestError,
    TeacherProxyTimeoutError,
)
from app.logging_utils import write_jsonl
from app.rag import RagService, build_context
from app.schemas import AskRequest, AskResponse, UploadRequest, UploadResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.rag_service = RagService(settings)
    app.state.llm_service = LlmService(settings)
    yield


app = FastAPI(title="RAG Competition Student Server", lifespan=lifespan)


@app.get("/")
def health_check():
    settings: Settings = app.state.settings
    rag_service: RagService = app.state.rag_service
    return {
        "status": "running",
        "student_id": settings.student_id,
        "teacher_base_url": settings.teacher_base_url,
        "llm_base_url": settings.teacher_proxy_base_url,
        "embedding_model_path": str(settings.embedding_model_path),
        "embedding_model_path_exists": settings.embedding_model_path.is_dir(),
        "vector_db_path": str(settings.vector_db_path),
        "vector_db_path_exists": settings.vector_db_path.is_file(),
        "log_path": str(settings.log_path),
        "chunks": rag_service.chunk_count,
    }


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
def upload(payload: UploadRequest) -> UploadResponse:
    settings: Settings = app.state.settings
    rag_service: RagService = app.state.rag_service

    try:
        doc_id, chunk_count = rag_service.ingest(payload.text, payload.doc_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    write_jsonl(
        settings.log_path,
        {
            "event": "upload",
            "doc_id": doc_id,
            "text_length": len(payload.text),
            "chunks": chunk_count,
        },
    )
    return UploadResponse(status="success", doc_id=doc_id, chunks=chunk_count)


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    settings: Settings = app.state.settings
    rag_service: RagService = app.state.rag_service
    llm_service: LlmService = app.state.llm_service

    try:
        scored_chunks = rag_service.retrieve(payload.question, settings.top_k)
    except LookupError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    context, selected_chunks = build_context(scored_chunks, settings.max_context_chars)
    raw_answer = ""
    llm_error = None

    try:
        llm_answer = llm_service.answer_question(payload.question, context)
        answer = llm_answer.answer
        raw_answer = llm_answer.raw_answer
    except TeacherProxyTimeoutError as error:
        write_jsonl(
            settings.log_path,
            {
                "event": "ask",
                "question": payload.question,
                "selected_chunk_ids": [item.chunk.chunk_id for item in scored_chunks[: len(selected_chunks)]],
                "selected_chunk_scores": [item.score for item in scored_chunks[: len(selected_chunks)]],
                "sources": [chunk.text for chunk in selected_chunks],
                "raw_answer": raw_answer,
                "answer": None,
                "llm_error": str(error),
            },
        )
        raise HTTPException(status_code=504, detail=str(error)) from error
    except (TeacherProxyRequestError, LlmAnswerParseError) as error:
        write_jsonl(
            settings.log_path,
            {
                "event": "ask",
                "question": payload.question,
                "selected_chunk_ids": [item.chunk.chunk_id for item in scored_chunks[: len(selected_chunks)]],
                "selected_chunk_scores": [item.score for item in scored_chunks[: len(selected_chunks)]],
                "sources": [chunk.text for chunk in selected_chunks],
                "raw_answer": raw_answer,
                "answer": None,
                "llm_error": str(error),
            },
        )
        raise HTTPException(status_code=502, detail=str(error)) from error

    write_jsonl(
        settings.log_path,
        {
            "event": "ask",
            "question": payload.question,
            "selected_chunk_ids": [item.chunk.chunk_id for item in scored_chunks[: len(selected_chunks)]],
            "selected_chunk_scores": [item.score for item in scored_chunks[: len(selected_chunks)]],
            "sources": [chunk.text for chunk in selected_chunks],
            "raw_answer": raw_answer,
            "answer": answer,
            "llm_error": llm_error,
        },
    )
    return AskResponse(answer=answer, sources=[chunk.text for chunk in selected_chunks])
