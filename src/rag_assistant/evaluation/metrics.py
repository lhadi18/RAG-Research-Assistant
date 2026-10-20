from math import log2


def recall_at_k(retrieved, relevant, k):
    relevant = set(relevant)
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def mrr_at_k(retrieved, relevant, k):
    relevant = set(relevant)
    return next((1 / rank for rank, item in enumerate(retrieved[:k], 1) if item in relevant), 0.0)


def ndcg_at_k(retrieved, relevant, k):
    relevant = set(relevant)
    dcg = sum(1 / log2(rank + 1) for rank, item in enumerate(retrieved[:k], 1) if item in relevant)
    ideal = sum(1 / log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0
