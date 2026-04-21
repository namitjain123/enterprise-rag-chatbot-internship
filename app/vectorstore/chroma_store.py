import chromadb
import uuid
from app.core.config import CHROMA_PATH, COLLECTION_NAME
from app.rag.embeddings import embed_texts, embed_query

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def index_chunks(chunks: list[str]) -> None:
    if not chunks:
        return

    embeddings = embed_texts(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )


def query_chunks(query: str, top_k: int = 3) -> list[str]:
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    if not results or "documents" not in results or not results["documents"]:
        return []

    return results["documents"][0]


def get_all_chunks() -> list[str]:
    results = collection.get(include=["documents"])
    if not results or "documents" not in results:
        return []
    return results["documents"]