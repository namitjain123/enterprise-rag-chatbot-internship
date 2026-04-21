from app.vectorstore.chroma_store import query_chunks
from app.rag.keyword_retriever import keyword_search


def hybrid_retrieve(query: str, vector_k: int = 5, keyword_k: int = 5) -> list[str]:
    vector_results = query_chunks(query=query, top_k=vector_k)
    keyword_results = keyword_search(query=query, top_k=keyword_k)

    combined = []
    seen = set()

    for chunk in vector_results + keyword_results:
        if chunk not in seen:
            combined.append(chunk)
            seen.add(chunk)

    return combined