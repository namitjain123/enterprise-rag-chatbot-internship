import re
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer

nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text: str) -> list[str]:
    text = clean_text(text)
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def semantic_chunk_text(
    text: str,
    similarity_threshold: float = 0.65,
    max_chunk_sentences: int = 5
) -> list[str]:
    """
    Split text into semantic chunks by grouping similar consecutive sentences.
    """
    sentences = split_into_sentences(text)

    if not sentences:
        return []

    if len(sentences) == 1:
        return sentences

    sentence_embeddings = embedder.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        prev_embedding = sentence_embeddings[i - 1]
        curr_embedding = sentence_embeddings[i]

        similarity = cosine_similarity(prev_embedding, curr_embedding)

        # keep adding if semantically similar and chunk not too large
        if similarity >= similarity_threshold and len(current_chunk) < max_chunk_sentences:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks