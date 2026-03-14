from app.vectorstore.chroma_store import query_chunks
from app.llm.groq_client import query_groq
from app.core.config import TOP_K


def retrieve_context(query: str, top_k: int = TOP_K) -> list[str]:
    return query_chunks(query=query, top_k=top_k)


def generate_rag_answer(query: str, top_k: int = TOP_K) -> dict:
    retrieved_chunks = retrieve_context(query, top_k=top_k)

    if not retrieved_chunks:
        return {
            "answer": "I could not find relevant information in the indexed documents.",
            "sources": []
        }

    context = "\n\n".join(retrieved_chunks)
    answer = query_groq(query, context)

    return {
        "answer": answer,
        "sources": retrieved_chunks
    }