from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from pathlib import Path


router = APIRouter()
templates = Jinja2Templates(directory="RAG_Novels/templates")
DEFAULT_DIRECTORY_PATH = str(Path(__file__).resolve().parent / "novels")


class IngestRequest(BaseModel):
    directory_path: str = DEFAULT_DIRECTORY_PATH


class AskRequest(BaseModel):
    question: str
    top_k: int = 3


@router.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/ingest")
def ingest_documents(payload: IngestRequest, request: Request):
    directory_path = Path(payload.directory_path).expanduser().resolve()
    if not directory_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Directory does not exist: {directory_path}",
        )

    rag_service = request.app.state.rag_service
    try:
        chunks = rag_service.ingest(str(directory_path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "message": "Ingestion completed",
        "chunks_count": len(chunks),
        "directory_path": str(directory_path),
    }


@router.post("/ask")
def ask_question(payload: AskRequest, request: Request):
    rag_service = request.app.state.rag_service
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    result = rag_service.ask(payload.question, top_k=payload.top_k)
    return {
        "question": payload.question,
        "answer": result["text"],
        "usage": result["usage"],
    }
