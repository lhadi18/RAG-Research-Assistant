from dataclasses import dataclass
from pathlib import Path
import hashlib


class IngestionError(ValueError):
    pass


@dataclass
class ParsedDocument:
    filename: str
    title: str
    content_hash: str
    pages: list[tuple[int | None, str]]


def parse_document(filename: str, content: bytes) -> ParsedDocument:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md", ".markdown"}:
        raise IngestionError("Unsupported file type. Upload a PDF, TXT, or Markdown file.")
    if not content:
        raise IngestionError("The uploaded file is empty.")
    digest = hashlib.sha256(content).hexdigest()
    title = Path(filename).stem
    if suffix == ".pdf":
        try:
            import fitz
            with fitz.open(stream=content, filetype="pdf") as pdf:
                title = (pdf.metadata or {}).get("title") or title
                pages = [(i + 1, page.get_text("text").strip()) for i, page in enumerate(pdf)]
        except Exception as exc:
            raise IngestionError("Could not read this PDF. It may be corrupted or encrypted.") from exc
        if not any(text for _, text in pages):
            raise IngestionError("This PDF has no extractable text. OCR is not supported.")
        return ParsedDocument(filename, title, digest, pages)
    try:
        text = content.decode("utf-8-sig").replace("\x00", "").strip()
    except UnicodeDecodeError as exc:
        raise IngestionError("Text documents must be UTF-8 encoded.") from exc
    if not text:
        raise IngestionError("The document contains no extractable text.")
    # Use the first Markdown heading as a human-readable title when available.
    if suffix in {".md", ".markdown"}:
        title = next((line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")), title)
    return ParsedDocument(filename, title, digest, [(None, text)])
