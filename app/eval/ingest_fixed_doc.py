"""
Ingest the single fixed evaluation PDF into ChromaDB.

This replaces the interactive upload flow for evaluation: instead of a user
uploading documents through the UI, we pin ONE known document so the RAG
pipeline and the RAGAS ground-truth answers always refer to the same source.

Run once before evaluating:
    python -m app.eval.ingest_fixed_doc
"""

import os

import app.vectorstore.chroma_store as chroma_store
from app.core.config import COLLECTION_NAME
from app.rag.rag_pipeline import process_pdf
from app.vectorstore.chroma_store import client, index_chunks

# The fixed document every evaluation question is written against.
FIXED_PDF_PATH = os.path.join("docs", "Leave-Policy.pdf")


def reset_collection() -> None:
    """Drop and recreate the collection so eval always starts from a clean, known state."""
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")
    except Exception:
        # Collection may not exist yet - that is fine.
        print(f"No existing collection '{COLLECTION_NAME}' to delete.")

    # chroma_store cached a handle to the now-deleted collection at import time;
    # recreate it and refresh that module-level handle so index_chunks() works.
    chroma_store.collection = client.get_or_create_collection(name=COLLECTION_NAME)


def ingest_fixed_document() -> int:
    """Process and index the fixed PDF. Returns the number of chunks indexed."""
    if not os.path.exists(FIXED_PDF_PATH):
        raise FileNotFoundError(
            f"Fixed PDF not found at '{FIXED_PDF_PATH}'. "
            "Run this from the project root (enterprise-rag-chatbot-main)."
        )

    chunks = process_pdf(FIXED_PDF_PATH)
    index_chunks(chunks)
    return len(chunks)


def main() -> None:
    reset_collection()
    n = ingest_fixed_document()
    print(f"Indexed {n} chunks from '{FIXED_PDF_PATH}' into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
