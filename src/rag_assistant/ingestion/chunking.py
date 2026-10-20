import hashlib
import re
import uuid
from rag_assistant.schemas import Chunk


def _tokens(text: str) -> list[str]:
    # Approximate token boundaries without an external tokenizer; configurable units are words.
    return re.findall(r"\S+", text)


def chunk_pages(pages, document_id: str, filename: str, title: str, content_hash: str,
                size: int = 800, overlap: int = 120) -> list[Chunk]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Chunk size must be positive and overlap smaller than size.")
    chunks = []
    position = 0
    for page, text in pages:
        words = _tokens(text)
        step = size - overlap
        for start in range(0, len(words), step):
            part = " ".join(words[start:start + size]).strip()
            if not part:
                continue
            raw_id = f"{document_id}:{page or 0}:{position}:{part}"
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, raw_id))
            chunks.append(Chunk(chunk_id=chunk_id, document_id=document_id, filename=filename,
                                title=title, page=page, position=position, text=part,
                                content_hash=content_hash))
            position += 1
            if start + size >= len(words):
                break
    return chunks
