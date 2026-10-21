# Evaluation dataset

`golden_dataset.jsonl` is version 1. Judgments identify relevant documents by stable content-derived document ID and refer to sources that state the described fact. Empty relevance sets are unanswerable items; they remain visible and receive zero in retrieval averages. All chunks from a relevant document currently count as relevant, so passage relevance is coarse. The corpus is self-authored demonstration material, not published research. Retrieval metrics are deterministic; no answer correctness is implied.
