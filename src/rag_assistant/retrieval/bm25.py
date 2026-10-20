import re
from collections import Counter
from math import log
from rag_assistant.schemas import Chunk


def terms(text: str) -> list[str]:
    return re.findall(r"[\w-]+", text.lower())


class BM25Retriever:
    """Use maintained rank-bm25 when installed, with an identical API for the local fallback."""
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.docs = [terms(c.text) for c in chunks]
        try:
            from rank_bm25 import BM25Okapi
            self.index = BM25Okapi(self.docs) if self.docs else None
        except ImportError:
            self.index = None
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        self.df = Counter(t for doc in self.docs for t in set(doc))

    def search(self, query: str, k: int = 5) -> list[Chunk]:
        if self.index is not None:
            scores = self.index.get_scores(terms(query))
            ranked = sorted(zip(scores, self.chunks), key=lambda x: float(x[0]), reverse=True)[:k]
            return [c.model_copy(update={"score": float(s)}) for s, c in ranked if s > 0]
        q = set(terms(query)); n = len(self.docs); scored = []
        for chunk, doc in zip(self.chunks, self.docs):
            counts = Counter(doc); score = 0.0
            for term in q:
                freq = counts[term]
                if not freq: continue
                idf = log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                score += idf * freq * 2.2 / (freq + 1.2 * (0.25 + 0.75 * len(doc) / max(self.avgdl, 1)))
            scored.append((score, chunk))
        return [c.model_copy(update={"score": float(s)}) for s, c in sorted(scored, key=lambda x: x[0], reverse=True)[:k] if s > 0]
