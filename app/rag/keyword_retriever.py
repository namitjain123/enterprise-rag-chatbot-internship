from rank_bm25 import BM25Okapi
from app.vectorstore.chroma_store import get_all_chunks


def keyword_search(query: str, top_k: int = 5) -> list[str]:
    chunks = get_all_chunks()

    if not chunks:
        return []

    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, score in scored_chunks[:top_k] if score > 0]