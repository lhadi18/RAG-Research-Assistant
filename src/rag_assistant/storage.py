"""Persistent local metadata plus optional Qdrant vector storage."""
import json
import hashlib
from pathlib import Path
from rag_assistant.schemas import Chunk


class DocumentStore:
    def __init__(self, data_dir: Path, settings):
        self.data_dir = Path(data_dir); self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "chunks.json"
        collection_key = hashlib.sha256(settings.qdrant_collection.encode()).hexdigest()[:16]
        self.vectors_path = self.data_dir / f"vectors_{collection_key}.json"
        self.settings = settings
        self.chunks = [Chunk.model_validate(x) for x in json.loads(self.path.read_text())] if self.path.exists() else []
        self.vectors = json.loads(self.vectors_path.read_text()) if self.vectors_path.exists() else {}
        self.qdrant = None
        try:
            from qdrant_client import QdrantClient
            self.qdrant = QdrantClient(url=settings.qdrant_url, timeout=2)
        except Exception:
            pass

    def save(self): self.path.write_text(json.dumps([c.model_dump() for c in self.chunks], ensure_ascii=False))
    def save_vectors(self): self.vectors_path.write_text(json.dumps(self.vectors))
    def add(self, chunks):
        existing = {c.chunk_id for c in self.chunks}
        self.chunks.extend(c for c in chunks if c.chunk_id not in existing)
        self.save()
    def delete(self, document_id):
        self.chunks = [c for c in self.chunks if c.document_id != document_id]; self.save()
        removed = [cid for cid, value in self.vectors.items() if value.get("document_id") == document_id]
        for cid in removed: self.vectors.pop(cid, None)
        self.save_vectors()
        if self.qdrant:
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                self.qdrant.delete(self.settings.qdrant_collection, Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]))
            except Exception: pass
    def documents(self):
        docs = {}
        for c in self.chunks:
            d = docs.setdefault(c.document_id, {"document_id": c.document_id, "filename": c.filename, "title": c.title, "chunks": 0, "status": "indexed", "content_hash": c.content_hash})
            d["chunks"] += 1
        return list(docs.values())

    def qdrant_ready(self):
        try: return bool(self.qdrant and self.qdrant.get_collections())
        except Exception: return False

    def index_vectors(self, chunks, vectors):
        existing_models = {item.get("embedding_model") for item in self.vectors.values() if item.get("embedding_model")}
        if existing_models and existing_models != {self.settings.ollama_embed_model}:
            raise ValueError("Existing vectors use a different embedding model. Delete/reindex the full collection or configure a new collection before indexing.")
        for chunk, vector in zip(chunks, vectors):
            self.vectors[chunk.chunk_id] = {"document_id": chunk.document_id, "embedding_model": self.settings.ollama_embed_model, "vector": vector}
        self.save_vectors()
        if self.qdrant:
            try:
                from qdrant_client.models import Distance, VectorParams, PointStruct
                name = self.settings.qdrant_collection
                existing = {c.name for c in self.qdrant.get_collections().collections}
                if name not in existing:
                    self.qdrant.create_collection(name, vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE))
                else:
                    info = self.qdrant.get_collection(name)
                    size = info.config.params.vectors.size
                    if size != len(vectors[0]):
                        raise ValueError("Qdrant collection dimensions differ from this embedding model. Use a new collection or fully reindex.")
                points = [PointStruct(id=c.chunk_id, vector=v, payload={**c.model_dump(exclude={"text", "score"}), "text": c.text}) for c, v in zip(chunks, vectors)]
                self.qdrant.upsert(name, points)
            except ValueError: raise
            except Exception as exc: raise RuntimeError("Could not write vectors to Qdrant. Check QDRANT_URL and service health.") from exc
