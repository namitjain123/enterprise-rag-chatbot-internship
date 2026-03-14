from sentence_transformers import CrossEncoder

_reranker_model = None


def get_reranker():
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_model


def rerank_chunks(query: str, chunks: list[str], top_n: int = 3) -> list[str]:
    """
    Rerank retrieved chunks using a cross-encoder model.
    """
    if not chunks:
        return []

    model = get_reranker()

    pairs = [[query, chunk] for chunk in chunks]
    scores = model.predict(pairs)

    scored_chunks = list(zip(chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in scored_chunks[:top_n]]