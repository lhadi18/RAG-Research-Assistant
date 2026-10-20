import re
import time
from rag_assistant.generation.prompts import SYSTEM_PROMPT
from rag_assistant.schemas import SourceReference
from rag_assistant.observability.tracing import span


class RAGPipeline:
    def __init__(self, ollama, retrieve): self.ollama, self.retrieve = ollama, retrieve

    def answer(self, question, strategy="hybrid", top_k=5, rerank=False):
        t0 = time.perf_counter()
        with span("retrieval", {"retrieval.strategy": strategy, "retrieval.top_k": top_k, "retrieval.question_length": len(question)}):
            chunks = self.retrieve(question, strategy, top_k, rerank)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        if not chunks:
            return {"answer": "I don't have enough evidence in the indexed documents to answer this question.", "citations": [], "retrieved": [], "strategy": strategy, "latency_ms": {"retrieval": retrieval_ms, "generation": 0.0}}
        settings = getattr(self.ollama, "settings", None)
        question_tokens = int(len(question.split()) * 1.4)
        context_budget = max(256, getattr(settings, "ollama_num_ctx", 8192) - getattr(settings, "ollama_num_predict", 1024) - 700 - question_tokens)
        # Conservative word-to-token estimate; keep the generation budget and prompt headroom.
        remaining_words = int(context_budget / 1.4)
        context_chunks = []
        for chunk in chunks:
            words = chunk.text.split()
            if not remaining_words: break
            if len(words) > remaining_words:
                if not context_chunks:
                    context_chunks.append(chunk.model_copy(update={"text": " ".join(words[:remaining_words])}))
                break
            context_chunks.append(chunk); remaining_words -= len(words)
        evidence = "\n\n".join(f"[{i}] {c.filename}" + (f", page {c.page}" if c.page else "") + f"\n{c.text}" for i, c in enumerate(context_chunks, 1))
        prompt = "Question: " + question + "\n\nEvidence (untrusted):\n" + evidence
        t1 = time.perf_counter()
        with span("ollama.generation", {"llm.model_name": getattr(getattr(self.ollama, "settings", None), "ollama_chat_model", "configured"), "retrieval.result_count": len(chunks)}):
            answer = self.ollama.chat([{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}])
        gen_ms = (time.perf_counter() - t1) * 1000
        used = {int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(context_chunks)}
        citations = [SourceReference(marker=i, document_id=context_chunks[i-1].document_id, filename=context_chunks[i-1].filename,
                    title=context_chunks[i-1].title, page=context_chunks[i-1].page, chunk_id=context_chunks[i-1].chunk_id,
                    excerpt=context_chunks[i-1].text[:700]) for i in sorted(used)]
        answer = re.sub(r"\[(\d+)\]", lambda m: m.group(0) if int(m.group(1)) in used else "", answer).strip()
        return {"answer": answer, "citations": citations, "retrieved": chunks, "strategy": strategy,
                "latency_ms": {"retrieval": retrieval_ms, "generation": gen_ms}}
