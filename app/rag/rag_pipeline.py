import fitz
from app.preprocessing.text_preprocessor import semantic_chunk_text


def extract_text_from_pdf(file_path: str) -> str:
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text


def process_pdf(file_path: str) -> list[str]:
    raw_text = extract_text_from_pdf(file_path)
    chunks = semantic_chunk_text(raw_text)
    return chunks