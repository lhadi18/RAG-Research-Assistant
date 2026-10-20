import json
import sys
import types
import pytest
from rag_assistant.ingestion.parsers import parse_document, IngestionError
from rag_assistant.ingestion.chunking import chunk_pages
from rag_assistant.retrieval.bm25 import BM25Retriever
from rag_assistant.retrieval.dense import DenseRetriever
from rag_assistant.retrieval.hybrid import reciprocal_rank_fusion
from rag_assistant.evaluation.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from rag_assistant.generation.rag_pipeline import RAGPipeline
from rag_assistant.schemas import Chunk
from rag_assistant.storage import DocumentStore
from rag_assistant.config import Settings


def chunk(cid, text, doc="doc"):
    return Chunk(chunk_id=cid, document_id=doc, filename=doc+".txt", position=0, text=text, content_hash="hash")


def test_parse_text_and_markdown_title():
    parsed=parse_document("x.md", b"# Clear Title\n\ntext")
    assert parsed.title == "Clear Title" and parsed.pages[0][1].endswith("text")


def test_parse_pdf_page_metadata():
    import fitz
    pdf=fitz.open(); page=pdf.new_page(); page.insert_text((72,72),"PDF evidence passage")
    data=pdf.tobytes(); pdf.close()
    parsed=parse_document("source.pdf",data)
    assert parsed.pages[0][0]==1 and "PDF evidence" in parsed.pages[0][1]


@pytest.mark.parametrize("name,data", [("a.csv",b"x"),("a.txt",b""),("a.pdf",b"garbage")])
def test_bad_documents(name,data):
    with pytest.raises(IngestionError): parse_document(name,data)


def test_deterministic_chunks_and_overlap():
    pages=[(2,"one two three four five six")]
    a=chunk_pages(pages,"d","a.txt","A","h",4,1)
    b=chunk_pages(pages,"d","a.txt","A","h",4,1)
    assert [x.chunk_id for x in a] == [x.chunk_id for x in b]
    assert len(a)==2 and a[0].page==2


def test_bm25_and_hybrid():
    a,b=chunk("a","qdrant vector search"),chunk("b","garden soil plants")
    found=BM25Retriever([a,b]).search("vector search",1)
    assert found[0].chunk_id=="a"
    fused=reciprocal_rank_fusion([[a,b],[b,a]],k=60,top_k=2)
    assert {c.chunk_id for c in fused}=={"a","b"}


def test_dense_with_mock_embedder():
    class Embedder:
        def embed(self,texts): return [[1,0] if "cat" in t else [0,1] for t in texts]
    result=DenseRetriever([chunk("a","cat"),chunk("b","dog")],Embedder()).search("cat",1)
    assert result[0].chunk_id=="a"


def test_store_duplicate_chunk_dedup(tmp_path):
    store=DocumentStore(tmp_path,Settings(data_dir=tmp_path))
    item=chunk("stable","same text")
    store.add([item]); store.add([item])
    assert len(store.chunks)==1


def test_store_rejects_mixed_embedding_models(tmp_path):
    settings=Settings(data_dir=tmp_path,ollama_embed_model="new-model")
    store=DocumentStore(tmp_path,settings)
    store.vectors["old"]={"document_id":"old-doc","embedding_model":"old-model","vector":[1.0]}
    with pytest.raises(ValueError,match="different embedding model"):
        store.index_vectors([chunk("new","text")],[[1.0]])


def test_local_reranker_uses_model_scores(monkeypatch):
    class FakeCrossEncoder:
        def __init__(self, model): self.model=model
        def predict(self, pairs): return [0.2 if "first" in text else 0.9 for _,text in pairs]
    monkeypatch.setitem(sys.modules,"sentence_transformers",types.SimpleNamespace(CrossEncoder=FakeCrossEncoder))
    from rag_assistant.retrieval.reranking import LocalReranker
    result=LocalReranker("mock/local").rerank("q",[chunk("a","first"),chunk("b","second")],1)
    assert result[0].chunk_id=="b"


def test_metrics_hand_calculated():
    assert recall_at_k(["x","a","b"],["a","b"],2)==.5
    assert mrr_at_k(["x","a"],["a"],2)==.5
    assert ndcg_at_k(["a","x","b"],["a","b"],3)==pytest.approx((1+1/2)/(1+1/1.5849625007))


def test_rag_citations_and_abstention():
    class Model:
        def chat(self,messages): return "Supported claim [1], fake reference [9]."
    result=RAGPipeline(Model(),lambda *args:[chunk("a","proof")]).answer("question")
    assert "[1]" in result["answer"] and "[9]" not in result["answer"]
    empty=RAGPipeline(Model(),lambda *args:[]).answer("question")
    assert "enough evidence" in empty["answer"]


def test_evaluation_result_json_serialization():
    from rag_assistant.evaluation.metrics import recall_at_k
    result={"strategy":"bm25","recall@5":recall_at_k(["a"],["a"],5),"failures":[]}
    assert json.loads(json.dumps(result))["recall@5"]==1.0
