# Local Retrieval-Augmented Generation: Design Note (Demonstration Corpus)

This self-authored teaching note is invented for software demonstration. It is not a scientific publication.

## Retrieval and generation
Retrieval-Augmented Generation (RAG) retrieves passages from a collection and supplies them to a language model as context. Dense retrieval compares vector embeddings and can match related phrasing. Lexical retrieval such as BM25 scores term overlap. Hybrid retrieval combines complementary rankings; Reciprocal Rank Fusion (RRF) adds a rank-based contribution of 1 divided by (k plus rank) for each result list.

## Chunking and evidence
Chunk size and overlap affect both retrieval precision and context coverage. Small chunks can omit context; large chunks can include irrelevant material. Overlap can preserve facts split at boundaries but increases index size and duplicate evidence. Stable chunk identifiers and source metadata support reproducibility and citation validation.

## Grounded answers
A grounded system should treat retrieved text as untrusted input, cite only passages actually supplied to generation, and abstain when the evidence does not answer the question. Model-produced filenames and page numbers should not be trusted; citations should be resolved against retrieved chunk records. Contradictory passages should be surfaced rather than collapsed into false certainty.

## Evaluation
Recall at k is the fraction of judged relevant passages present in the first k retrieved results. Reciprocal rank is the inverse rank of the first relevant result, or zero when none appears within the cutoff. nDCG uses graded or binary relevance with logarithmic rank discount, normalized by the ideal ranking. Metrics depend on careful relevance judgments and do not prove that generated answers are correct.

## Local inference and privacy
When models run through a local Ollama service, inference can stay on the user's computer. Model downloads still require disk space and often an initial internet connection. Embedding indexes are tied to an embedding model and vector dimension; changing either generally requires a full reindex. Local execution does not remove the need to protect uploaded files, logs, backups, and the host machine.
