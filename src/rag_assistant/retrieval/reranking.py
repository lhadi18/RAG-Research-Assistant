from rag_assistant.observability.tracing import span


class LocalReranker:
    def __init__(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Install optional reranker dependencies: pip install -e '.[rerank]'") from exc
        # sentence-transformers downloads this model on explicit use, never at application startup.
        self.model = CrossEncoder(model_name)

    def rerank(self, query, chunks, k=5):
        if not chunks: return []
        with span("local.reranking", {"reranker.candidate_count": len(chunks)}):
            scores = self.model.predict([(query, c.text) for c in chunks])
        ordered = sorted(zip(scores, chunks), key=lambda x: float(x[0]), reverse=True)[:k]
        return [c.model_copy(update={"score": float(s)}) for s, c in ordered]
from rag_assistant.observability.tracing import span

