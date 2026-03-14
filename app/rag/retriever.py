from app.llm.groq_client import query_groq
from app.core.config import TOP_K
from app.rag.reranker import rerank_chunks
from app.rag.hybrid_retriever import hybrid_retrieve


def generate_rag_answer(query: str, top_k: int = TOP_K) -> dict:
    # Step 1: hybrid retrieval
    retrieved_chunks = hybrid_retrieve(query=query, vector_k=8, keyword_k=8)

    if not retrieved_chunks:
        return {
            "answer": "I could not find relevant information in the indexed documents.",
            "sources": []
        }

    print("\n--- Hybrid Retrieved Chunks ---")
    for i, chunk in enumerate(retrieved_chunks[:5], start=1):
        print(f"{i}. {chunk[:200]}")

    # Step 2: rerank the combined candidates
    best_chunks = rerank_chunks(query, retrieved_chunks, top_n=top_k)

    print("\n--- Reranked Best Chunks ---")
    for i, chunk in enumerate(best_chunks, start=1):
        print(f"{i}. {chunk[:200]}")

    # Step 3: final answer generation
    context = "\n\n".join(best_chunks)
    answer = query_groq(query, context)

    return {
        "answer": answer,
        "sources": best_chunks
    }