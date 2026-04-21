from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import tempfile
from typing import List

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
            "chunks_indexed": total_chunks
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