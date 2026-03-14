from sentence_transformers import SentenceTransformer
from app.core.config import EMBEDDING_MODEL

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return model.encode(texts).tolist()


def embed_query(query: str) -> list[float]:
    model = get_embedding_model()
    return model.encode(query).tolist()