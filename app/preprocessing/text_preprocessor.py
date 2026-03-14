import re
import spacy
from app.core.config import CHUNK_SIZE, CHUNK_OVERLAP

nlp = spacy.load("en_core_web_sm")


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_into_sentences(text: str) -> list[str]:
    text = clean_text(text)
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def make_sentence_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            if overlap > 0 and chunks:
                tail = current_chunk[-overlap:]
                current_chunk = (tail + " " + sentence).strip()
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks