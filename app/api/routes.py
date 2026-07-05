from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import tempfile
from typing import List

from app.rag.rag_pipeline import process_pdf
from app.vectorstore.chroma_store import index_chunks
from app.rag.guarded_pipeline import generate_guarded_answer
from app.core.config import GUARDRAILS_ENABLED

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


@router.get("/health")
def health_check():
    return {"status": "running"}


@router.post("/ingest")
async def ingest_pdfs(files: List[UploadFile] = File(...)):
    total_chunks = 0
    uploaded_files = []

    try:
        for file in files:
            if not file.filename.endswith(".pdf"):
                raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF.")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_path = temp_file.name

            chunks = process_pdf(temp_path)
            index_chunks(chunks)

            total_chunks += len(chunks)
            uploaded_files.append(file.filename)

            os.remove(temp_path)

        return {
            "message": "Documents uploaded and indexed successfully.",
            "files_uploaded": uploaded_files,
            "chunks_indexed": total_chunks,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
def query_document(request: QueryRequest):
    try:
        result = generate_guarded_answer(request.question, request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/query/stream")
def query_stream(question: str):
    """SSE endpoint — yields tokens as `data: <token>\\n\\n` so Gradio can stream them."""
    def event_generator():
        try:
            if GUARDRAILS_ENABLED:
                from app.guardrails.nemo_pipeline import generate_nemo_answer_stream
                for token in generate_nemo_answer_stream(question):
                    yield f"data: {token}\n\n"
            else:
                from app.rag.retriever import generate_rag_answer
                result = generate_rag_answer(question)
                yield f"data: {result['answer']}\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")