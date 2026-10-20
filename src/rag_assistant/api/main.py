import hashlib
import logging
import time
from fastapi import FastAPI, UploadFile, HTTPException, File, Query
from fastapi.responses import JSONResponse
from rag_assistant.config import get_settings
from rag_assistant.schemas import QueryRequest, QueryResponse, HealthResponse
from rag_assistant.storage import DocumentStore
from rag_assistant.ingestion.parsers import parse_document, IngestionError
from rag_assistant.ingestion.chunking import chunk_pages
from rag_assistant.generation.ollama_client import OllamaClient, OllamaError
from rag_assistant.retrieval.bm25 import BM25Retriever
from rag_assistant.retrieval.dense import DenseRetriever
from rag_assistant.retrieval.hybrid import HybridRetriever
from rag_assistant.retrieval.reranking import LocalReranker
from rag_assistant.generation.rag_pipeline import RAGPipeline
from rag_assistant.observability.tracing import initialize, span

logging.basicConfig(level=get_settings().log_level)
settings = get_settings()
store = DocumentStore(settings.data_dir, settings)
ollama = OllamaClient(settings)
app = FastAPI(title="Local RAG Research Assistant", version="0.1.0", description="Private, local-first document research with Ollama.")
initialize()


@app.exception_handler(OllamaError)
async def ollama_error_handler(_, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse)
def health():
    status = ollama.status(); ready = status.get("reachable") and status.get("chat_model") and status.get("embedding_model")
    return HealthResponse(status="ready" if ready else "degraded", ollama=bool(status.get("reachable")), qdrant=store.qdrant_ready(), detail="Ready" if ready else status.get("detail", "Ollama is reachable but configured model(s) are missing."))


@app.get("/health/ollama")
def health_ollama():
    status = ollama.status()
    status["setup"] = "Install/start Ollama, then run: ollama pull " + settings.ollama_chat_model + " && ollama pull " + settings.ollama_embed_model
    return status


@app.post("/documents/upload")
async def upload(files: list[UploadFile] = File(...)):
    if not files: raise HTTPException(400, "Select at least one document.")
    results = []
    for file in files:
        filename = (file.filename or "").replace("\\", "/").split("/")[-1]
        data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        if len(data) > settings.max_upload_mb * 1024 * 1024: results.append({"filename": filename, "status": "error", "error": f"File exceeds {settings.max_upload_mb} MB."}); continue
        try:
            parsed = parse_document(filename, data)
            if any(d["content_hash"] == parsed.content_hash for d in store.documents()):
                results.append({"filename": filename, "status": "duplicate", "error": "This document is already indexed."}); continue
            doc_id = parsed.content_hash[:32]
            chunks = chunk_pages(parsed.pages, doc_id, parsed.filename, parsed.title, parsed.content_hash, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
            chunks = [c.model_copy(update={"metadata": {"embedding_model": settings.ollama_embed_model,
                "chunk_size_tokens": settings.chunk_size_tokens, "chunk_overlap_tokens": settings.chunk_overlap_tokens,
                "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}) for c in chunks]
            with span("ollama.embeddings", {"embedding.model_name": settings.ollama_embed_model, "embedding.input_count": len(chunks)}):
                vectors = ollama.embed([c.text for c in chunks])
            store.index_vectors(chunks, vectors); store.add(chunks)
            results.append({"filename": filename, "status": "indexed", "document_id": doc_id, "chunks": len(chunks)})
        except (IngestionError, OllamaError, RuntimeError, ValueError) as exc:
            results.append({"filename": filename, "status": "error", "error": str(exc)})
    return {"results": results}


@app.get("/documents")
def documents(): return store.documents()


@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    if not any(d["document_id"] == document_id for d in store.documents()): raise HTTPException(404, "Document not found.")
    store.delete(document_id); return {"status": "deleted", "document_id": document_id}


@app.post("/documents/{document_id}/reindex")
def reindex_document(document_id: str):
    chunks = [c for c in store.chunks if c.document_id == document_id]
    if not chunks: raise HTTPException(404, "Document not found.")
    try:
        with span("ollama.embeddings", {"embedding.model_name": settings.ollama_embed_model, "embedding.input_count": len(chunks)}):
            vectors = ollama.embed([c.text for c in chunks])
        store.index_vectors(chunks, vectors)
        return {"status": "indexed", "document_id": document_id, "chunks": len(chunks), "embedding_model": settings.ollama_embed_model}
    except (OllamaError, RuntimeError, ValueError) as exc:
        raise HTTPException(503, str(exc)) from exc


def retrieve(question, strategy, top_k, rerank=False):
    chunks = store.chunks
    if strategy == "bm25": return BM25Retriever(chunks).search(question, top_k)
    dense = DenseRetriever(chunks, __import__("rag_assistant.embeddings.ollama_embeddings", fromlist=["OllamaEmbeddings"]).OllamaEmbeddings(ollama), store.vectors, store.qdrant, settings.qdrant_collection)
    if strategy == "dense": return dense.search(question, top_k)
    hybrid = HybridRetriever(BM25Retriever(chunks), dense, settings.rrf_k)
    if strategy == "hybrid_rerank" or rerank:
        try: return LocalReranker(settings.reranker_model).rerank(question, hybrid.search(question, settings.retrieval_candidates), top_k)
        except RuntimeError as exc: raise HTTPException(503, str(exc))
    return hybrid.search(question, top_k, settings.retrieval_candidates)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not store.chunks: raise HTTPException(400, "No documents are indexed yet.")
    result = RAGPipeline(ollama, retrieve).answer(request.question, request.strategy, request.top_k, request.rerank)
    return QueryResponse(**result)


@app.post("/eval/run")
def eval_run(generation: bool = Query(default=False)):
    from rag_assistant.evaluation.runner import run
    return run(include_generation=generation)


@app.get("/eval/results")
def eval_results():
    from pathlib import Path
    path = Path("reports/latest.json")
    if not path.exists(): raise HTTPException(404, "No evaluation results exist yet.")
    import json
    return json.loads(path.read_text())
