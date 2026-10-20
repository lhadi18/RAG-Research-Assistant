from rag_assistant.schemas import Chunk


def reciprocal_rank_fusion(rankings: list[list[Chunk]], k: int = 60, top_k: int = 5) -> list[Chunk]:
    scores = {}; objects = {}
    for ranking in rankings:
        seen = set()
        for rank, chunk in enumerate(ranking, 1):
            if chunk.chunk_id in seen: continue
            seen.add(chunk.chunk_id); scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (k + rank)
            objects[chunk.chunk_id] = chunk
    ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [objects[cid].model_copy(update={"score": scores[cid]}) for cid in ordered]


class HybridRetriever:
    def __init__(self, bm25, dense, rrf_k=60): self.bm25, self.dense, self.rrf_k = bm25, dense, rrf_k
    def search(self, query, k=5, candidates=20):
        return reciprocal_rank_fusion([self.bm25.search(query, candidates), self.dense.search(query, candidates)], self.rrf_k, k)
