import chromadb
from app.core.config import CHROMA_PATH, COLLECTION_NAME
from app.rag.embeddings import embed_texts, embed_query

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def index_chunks(chunks: list[str]) -> None:
    if not chunks:
        return

    embeddings = embed_texts(chunks)
    ids = [f"doc_{i}" for i in range(len(chunks))]

    existing = collection.get(include=[])
    existing_count = len(existing["ids"]) if existing and "ids" in existing else 0
    ids = [f"doc_{existing_count + i}" for i in range(len(chunks))]

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


def reset_collection():
    global collection
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)