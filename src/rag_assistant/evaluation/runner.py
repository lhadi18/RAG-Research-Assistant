"""Retrieval-only reproducible experiment; no Ollama calls are made."""
import hashlib
import json
import time
import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from rag_assistant.config import get_settings
from rag_assistant.evaluation.datasets import load_dataset
from rag_assistant.evaluation.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from rag_assistant.retrieval.bm25 import BM25Retriever
from rag_assistant.storage import DocumentStore


def run(include_generation=False):
    settings = get_settings(); store = DocumentStore(settings.data_dir, settings)
    chunks = store.chunks
    dataset_path = Path("evals/golden_dataset.jsonl")
    questions = load_dataset(dataset_path)
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    corpus_hash = hashlib.sha256("".join(sorted(c.content_hash for c in chunks)).encode()).hexdigest()
    bm25 = BM25Retriever(chunks)
    results = []
    methods = {"bm25": lambda q: bm25.search(q, 5)}
    try:
        from rag_assistant.generation.ollama_client import OllamaClient
        from rag_assistant.retrieval.dense import DenseRetriever
        dense = DenseRetriever(chunks, OllamaClient())
        from rag_assistant.retrieval.hybrid import HybridRetriever
        hybrid = HybridRetriever(bm25, dense, settings.rrf_k)
        methods.update({"dense": lambda q: dense.search(q, 5), "hybrid": lambda q: hybrid.search(q, 5)})
        try:
            from rag_assistant.retrieval.reranking import LocalReranker
            reranker = LocalReranker(settings.reranker_model)
            methods["hybrid_rerank"] = lambda q: reranker.rerank(q, hybrid.search(q, settings.retrieval_candidates), 5)
        except Exception:
            pass
    except Exception:
        pass
    for name in ["bm25", "dense", "hybrid", "hybrid_rerank"]:
        if name not in methods:
            results.append({"strategy": name, "status": "skipped", "reason": "Local embedding service/model or optional reranker unavailable"}); continue
        scores = {"recall@5": [], "mrr@5": [], "ndcg@5": [], "latency_ms": []}; failures = []
        for item in questions:
            try:
                start = time.perf_counter(); found = methods[name](item["question"]); latency = (time.perf_counter()-start)*1000
                ids = [c.chunk_id for c in found]
                rel = item.get("relevant_chunk_ids", [])
                rel += [c.chunk_id for c in chunks if c.document_id in item.get("relevant_document_ids", [])]
                scores["recall@5"].append(recall_at_k(ids, rel, 5)); scores["mrr@5"].append(mrr_at_k(ids, rel, 5)); scores["ndcg@5"].append(ndcg_at_k(ids, rel, 5)); scores["latency_ms"].append(latency)
            except Exception as exc: failures.append({"question_id": item["id"], "error": str(exc)})
        results.append({"strategy": name, "status": "complete", "questions": len(questions), "metrics": {key: mean(values) if values else 0 for key, values in scores.items()}, "failures": failures})
    answer_evaluation = None
    if include_generation:
        if "hybrid" not in methods:
            answer_evaluation = {"status": "skipped", "reason": "Hybrid retrieval needs the configured local Ollama embedding model."}
        else:
            from rag_assistant.generation.rag_pipeline import RAGPipeline
            from rag_assistant.generation.ollama_client import OllamaClient
            generated = []; failures = []; generation_ms = []; citation_valid = []; evidence_coverage = []; abstention_correct = []
            pipeline = RAGPipeline(OllamaClient(), lambda q, strategy, top_k, rerank: methods["hybrid"](q))
            for item in questions:
                try:
                    start = time.perf_counter()
                    answer = pipeline.answer(item["question"], "hybrid", 5)
                    generation_ms.append(answer["latency_ms"]["generation"])
                    retrieved_ids = {c.chunk_id for c in answer["retrieved"]}
                    relevant_ids = {c.chunk_id for c in chunks if c.document_id in item.get("relevant_document_ids", [])}
                    evidence_coverage.append(len(retrieved_ids & relevant_ids) / len(relevant_ids) if relevant_ids else 0.0)
                    valid = all(c.chunk_id in retrieved_ids for c in answer["citations"])
                    citation_valid.append(float(valid))
                    text = answer["answer"].lower()
                    abstains = any(term in text for term in ("not enough evidence", "insufficient evidence", "cannot answer", "can't answer"))
                    if not item.get("answerable", True): abstention_correct.append(float(abstains))
                    generated.append({"question_id": item["id"], "answer": answer["answer"], "citation_valid": valid, "citations": [c.model_dump() for c in answer["citations"]], "latency_ms": answer["latency_ms"]})
                except Exception as exc: failures.append({"question_id": item["id"], "error": str(exc)})
            answer_evaluation = {"status": "complete", "model": settings.ollama_chat_model,
                "metrics": {"citation_validity": mean(citation_valid) if citation_valid else 0.0,
                "expected_evidence_coverage": mean(evidence_coverage) if evidence_coverage else 0.0,
                "unanswerable_abstention_accuracy": mean(abstention_correct) if abstention_correct else 0.0,
                "generation_latency_ms": mean(generation_ms) if generation_ms else 0.0},
                "token_statistics": "Unavailable from the current Ollama HTTP wrapper.", "failures": failures, "cases": generated}
    output = {"experiment_id": datetime.now(timezone.utc).strftime("retrieval-%Y%m%dT%H%M%SZ"), "timestamp": datetime.now(timezone.utc).isoformat(), "dataset_version": "1", "dataset_hash": dataset_hash, "corpus_hash": corpus_hash, "embedding_model": settings.ollama_embed_model, "reranker_model": settings.reranker_model, "reranker_enabled": settings.reranker_enabled, "top_k": 5, "results": results, "answer_evaluation": answer_evaluation}
    Path("reports").mkdir(exist_ok=True); Path("reports/latest.json").write_text(json.dumps(output, indent=2))
    try:
        import pandas as pd
        pd.DataFrame([{"strategy": r["strategy"], "status": r["status"], **r.get("metrics", {})} for r in results]).to_csv("reports/latest.csv", index=False)
        complete = [r for r in results if r["status"] == "complete"]
        if complete:
            if os.getenv("RAG_SKIP_MATPLOTLIB") != "1":
                cache = Path("reports/.plot_cache").resolve(); cache.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("MPLCONFIGDIR", str(cache))
                os.environ.setdefault("XDG_CACHE_HOME", str(cache))
                import matplotlib.pyplot as plt
                frame = pd.DataFrame([{"strategy": r["strategy"], **r["metrics"]} for r in complete]).set_index("strategy")
                ax = frame[["recall@5", "mrr@5", "ndcg@5"]].plot(kind="bar", ylim=(0, 1), figsize=(9, 5))
                ax.set_ylabel("Score"); ax.set_title("Retrieval metrics (measured run)"); plt.tight_layout(); plt.savefig("reports/latest.png"); plt.close()
            else:
                _write_svg_plot(complete)
    except Exception:
        complete = [r for r in results if r["status"] == "complete"]
        if complete: _write_svg_plot(complete)
    print(json.dumps(output, indent=2))
    return output


def _write_svg_plot(results):
    """Portable SVG fallback for restricted systems where native Matplotlib rendering fails."""
    colors = ["#2563eb", "#16a34a", "#f59e0b"]
    metrics = ["recall@5", "mrr@5", "ndcg@5"]
    width, height, base_y, plot_h = 780, 360, 280, 230
    items = ['<svg xmlns="http://www.w3.org/2000/svg" width="780" height="360" viewBox="0 0 780 360">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="390" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">Retrieval metrics (measured run)</text>']
    for tick in range(0, 6):
        y = base_y - tick * plot_h / 5; value = tick / 5
        items.append(f'<line x1="80" y1="{y:.1f}" x2="750" y2="{y:.1f}" stroke="#e5e7eb"/>')
        items.append(f'<text x="68" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.1f}</text>')
    for i, result in enumerate(results):
        x0 = 145 + i * 170
        for j, metric in enumerate(metrics):
            value = max(0.0, min(1.0, float(result["metrics"].get(metric, 0))))
            bar_h = value * plot_h; x = x0 + j * 33
            items.append(f'<rect x="{x}" y="{base_y-bar_h:.1f}" width="25" height="{bar_h:.1f}" fill="{colors[j]}"/>')
        items.append(f'<text x="{x0+30}" y="305" text-anchor="middle" font-family="sans-serif" font-size="12">{result["strategy"]}</text>')
    for j, metric in enumerate(metrics):
        x = 160 + j * 170
        items.append(f'<rect x="{x}" y="330" width="12" height="12" fill="{colors[j]}"/><text x="{x+18}" y="341" font-family="sans-serif" font-size="12">{metric}</text>')
    items.append("</svg>")
    Path("reports/latest.svg").write_text("\n".join(items))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", action="store_true", help="Also evaluate answers using the local Ollama chat model (slower).")
    run(include_generation=parser.parse_args().generation)
