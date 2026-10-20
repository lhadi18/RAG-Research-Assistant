from rag_assistant.schemas import Chunk
from rag_assistant.observability.tracing import span


class DenseRetriever:
    def __init__(self, chunks: list[Chunk], embedder, stored_vectors=None, qdrant=None, collection=None):
        self.chunks, self.embedder, self.qdrant, self.collection = chunks, embedder, qdrant, collection
        self.by_id = {c.chunk_id: c for c in chunks}
        expected_model = getattr(getattr(embedder, "client", None), "settings", None)
        expected_model = getattr(expected_model, "ollama_embed_model", None)
        saved_models = {stored_vectors[c.chunk_id].get("embedding_model") for c in chunks
                        if stored_vectors and c.chunk_id in stored_vectors and stored_vectors[c.chunk_id].get("embedding_model")}
        if expected_model and saved_models and saved_models != {expected_model}:
            raise ValueError("Indexed vectors use a different embedding model; reindex all documents or use a separate Qdrant collection.")
        self.vectors = [stored_vectors[c.chunk_id]["vector"] for c in chunks if stored_vectors and c.chunk_id in stored_vectors]
        self.qdrant_complete = bool(expected_model and len(self.vectors) == len(chunks) and
                                    all(stored_vectors[c.chunk_id].get("embedding_model") == expected_model for c in chunks))
        if len(self.vectors) != len(chunks):
            self.vectors = embedder.embed([c.text for c in chunks]) if chunks else []

    @staticmethod
    def _cosine(a, b):
        dot = sum(x*y for x, y in zip(a, b)); na = sum(x*x for x in a) ** .5; nb = sum(y*y for y in b) ** .5
        return dot / (na * nb) if na and nb else 0.0

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        with span("ollama.embedding_query", {"embedding.input_count": 1}):
            vector = self.embedder.embed([query])[0]
        if self.qdrant is not None and self.collection and self.qdrant_complete:
            try:
                with span("qdrant.search", {"retrieval.top_k": k, "retrieval.backend": "qdrant"}):
                    response = self.qdrant.query_points(collection_name=self.collection, query=vector, limit=k, with_payload=True)
                result = []
                for point in response.points:
                    payload = point.payload or {}
                    chunk = self.by_id.get(str(point.id))
                    if chunk:
                        result.append(chunk.model_copy(update={"score": float(point.score or 0.0)}))
                if result: return result
            except Exception:
                # Local persisted vectors remain a usable offline fallback when Qdrant is stopped.
                pass
        if self.vectors and len(vector) != len(self.vectors[0]):
            raise ValueError("Query and index embedding dimensions differ; reindex with the same model.")
        scored = [(self._cosine(vector, v), c) for c, v in zip(self.chunks, self.vectors)]
        return [c.model_copy(update={"score": float(s)}) for s, c in sorted(scored, key=lambda x: x[0], reverse=True)[:k]]
