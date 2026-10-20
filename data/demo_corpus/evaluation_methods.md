# Practical Retrieval Evaluation Notes (Demonstration Corpus)

This self-authored note is invented for software demonstration. It is not a scientific publication.

## Benchmark construction
A retrieval benchmark pairs questions with passages that directly support an answer. Include single-passage, multi-passage, keyword-heavy, semantic, ambiguous, and unanswerable questions. Relevance judgments should be checked against the source text and saved with a dataset version or hash. Unanswerable examples should not assign arbitrary relevant passages.

## Comparison protocol
Compare BM25, dense retrieval, hybrid RRF, and hybrid plus reranking over the same corpus, chunks, questions, and relevance labels. Report the cutoff, chunking configuration, embedding model, reranker, latency, failures, and skipped runs. A reranker may improve ordering on a particular dataset, but that outcome must be measured rather than assumed.

## Answer evaluation
Citation validity checks whether every emitted marker maps to retrieved evidence. Expected-evidence coverage measures how much judged supporting evidence was included. Abstention checks whether the system declines questions without corpus support. A locally prompted judge can estimate faithfulness or relevance, but judge outputs are not ground truth and should be reported separately from deterministic metrics.
