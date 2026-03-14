from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import tempfile

from app.rag.rag_pipeline import process_pdf
from app.vectorstore.chroma_store import index_chunks
from app.rag.retriever import generate_rag_answer

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


@router.get("/health")
def health_check():
    return {"status": "running"}


@router.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        chunks = process_pdf(temp_path)
        index_chunks(chunks)

        os.remove(temp_path)

        return {
            "message": "Document uploaded and indexed successfully.",
            "chunks_indexed": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query")
def query_document(request: QueryRequest):
    try:
        result = generate_rag_answer(request.question, request.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))