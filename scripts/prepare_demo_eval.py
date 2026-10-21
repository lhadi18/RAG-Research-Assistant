"""Load demo text and chunks locally for BM25 evaluation without Ollama/Qdrant."""
from pathlib import Path
from rag_assistant.config import get_settings
from rag_assistant.ingestion.parsers import parse_document
from rag_assistant.ingestion.chunking import chunk_pages
from rag_assistant.storage import DocumentStore

settings = get_settings()
store = DocumentStore(settings.data_dir, settings)
all_chunks = []
for path in sorted(Path("data/demo_corpus").glob("*.md")):
    parsed = parse_document(path.name, path.read_bytes())
    doc_id = parsed.content_hash[:32]
    all_chunks.extend(chunk_pages(parsed.pages, doc_id, parsed.filename, parsed.title, parsed.content_hash,
                                  settings.chunk_size_tokens, settings.chunk_overlap_tokens))
store.add(all_chunks)
print(f"Prepared {len(all_chunks)} text chunks locally. No embeddings or model calls were made.")
