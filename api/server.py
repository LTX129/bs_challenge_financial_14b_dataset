import json
import os
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from framework import CHROMA_COLLECTION, CHROMA_PERSIST_PATH, CHROMA_SERVER_TYPE
from rag.rag import RagManager
from utils.util import get_model_bundle

QUESTION_PATH = Path(os.getenv("QUESTION_PATH", "question.json"))
WEB_DIR = Path(__file__).resolve().parent.parent / "webui"


class SourceSnippet(BaseModel):
    source: str
    preview: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=4000)
    question_id: Optional[int] = None
    backend: Optional[Literal["local", "qwen"]] = Field(
        default=None, description="Defaults to RAG_MODEL_BACKEND or local."
    )
    k: int = Field(default=4, ge=1, le=10, description="Top K results to retrieve.")
    mutuality: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Similarity score threshold."
    )


class AskResponse(BaseModel):
    answer: str
    backend: str
    question: str
    question_id: Optional[int] = None
    hits: int
    sources: list[SourceSnippet] = []


class QuestionItem(BaseModel):
    id: int
    question: str


class QuestionsResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[QuestionItem]


def _load_questions() -> list[dict]:
    """
    Load questions from question.json (supports jsonlines or JSON array).
    """
    if not QUESTION_PATH.exists():
        return []

    content = QUESTION_PATH.read_text(encoding="utf-8")
    content = content.strip()
    items: list[dict] = []

    if not content:
        return items

    try:
        if content.startswith("["):
            raw_items = json.loads(content)
            if isinstance(raw_items, dict):
                raw_items = [raw_items]
            for idx, item in enumerate(raw_items):
                if not isinstance(item, dict):
                    continue
                question_text = item.get("question") or item.get("prompt")
                if not question_text:
                    continue
                items.append({"id": item.get("id", idx), "question": question_text})
            return items
    except json.JSONDecodeError:
        # Fall back to jsonlines parsing
        pass

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue

        if not isinstance(obj, dict):
            continue
        question_text = obj.get("question") or obj.get("prompt")
        if not question_text:
            continue

        items.append({"id": obj.get("id", len(items)), "question": question_text})

    return items


QUESTIONS = _load_questions()


class RagService:
    """
    Manage RAG managers per backend so we can switch between local and remote LLM stacks.
    """

    def __init__(self) -> None:
        self.allowed_backends = {"local", "qwen"}
        default_backend = os.getenv("RAG_MODEL_BACKEND", "local").lower()
        self.default_backend = (
            default_backend if default_backend in self.allowed_backends else "local"
        )
        self._clients: dict[str, RagManager] = {}

    def _get_backend(self, backend: Optional[str]) -> str:
        selected = (backend or self.default_backend).lower()
        if selected not in self.allowed_backends:
            raise ValueError(
                f"Unsupported backend '{selected}'. Choose from {self.allowed_backends}"
            )
        return selected

    def _get_manager(self, backend: str) -> RagManager:
        if backend not in self._clients:
            llm, _chat, embed = get_model_bundle(prefer_backend=backend)
            self._clients[backend] = RagManager(
                chroma_server_type=CHROMA_SERVER_TYPE,
                persist_path=CHROMA_PERSIST_PATH,
                collection_name=CHROMA_COLLECTION,
                llm=llm,
                embed=embed,
            )
        return self._clients[backend]

    def ask(self, payload: AskRequest) -> dict:
        backend = self._get_backend(payload.backend)
        manager = self._get_manager(backend)
        result = manager.get_result_with_sources(
            question=payload.question.strip(),
            k=payload.k,
            mutuality=payload.mutuality,
        )
        result["backend"] = backend
        return result


rag_service = RagService()

app = FastAPI(title="Financial RAG QA", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
async def status():
    return {
        "backends": sorted(rag_service.allowed_backends),
        "default_backend": rag_service.default_backend,
        "question_total": len(QUESTIONS),
        "question_path": str(QUESTION_PATH),
        "collection_name": CHROMA_COLLECTION,
    }


@app.get("/api/questions", response_model=QuestionsResponse)
async def list_questions(
    q: Optional[str] = Query(default=None, description="Fuzzy search substring."),
    limit: int = Query(default=20, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
):
    filtered = QUESTIONS
    if q:
        query_lower = q.lower()
        filtered = [
            item for item in QUESTIONS if query_lower in item["question"].lower()
        ]

    total = len(filtered)
    items = filtered[offset : offset + limit]

    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.post("/api/ask", response_model=AskResponse)
async def ask(payload: AskRequest):
    text = payload.question.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Question is empty.")

    try:
        result = rag_service.ask(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "answer": result["answer"],
        "backend": result["backend"],
        "question": text,
        "question_id": payload.question_id,
        "hits": result.get("hits", 0),
        "sources": result.get("sources", []),
    }


@app.get("/", include_in_schema=False)
async def root():
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "RAG QA API is running.", "ui_missing": True}


if WEB_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="webui")
