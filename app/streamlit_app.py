import os
import base64
import requests
import pandas as pd
import streamlit as st

API = os.getenv("RAG_API_URL", "http://localhost:8000")
st.set_page_config(page_title="Local RAG Research Assistant", page_icon="📚", layout="wide")
st.title("Local RAG Research Assistant")
st.caption("Private document research powered by local Ollama inference")
try:
    health = requests.get(API + "/health/ollama", timeout=3).json()
    if not health.get("reachable") or not health.get("chat_model") or not health.get("embedding_model"):
        st.warning("Ollama or configured models are unavailable. " + health.get("setup", "Start Ollama and download the configured models."))
except requests.RequestException:
    st.warning("The API is unavailable. Start it with `uvicorn rag_assistant.api.main:app --reload`.")

tab_qa, tab_docs, tab_eval = st.tabs(["Research assistant", "Document library", "Evaluation"])
with tab_qa:
    strategy = st.selectbox("Retrieval strategy", ["bm25", "dense", "hybrid", "hybrid_rerank"], format_func=lambda x: {"bm25":"BM25", "dense":"Dense semantic", "hybrid":"Hybrid RRF", "hybrid_rerank":"Hybrid + local reranker"}[x])
    top_k = st.slider("Retrieved passages", 1, 15, 5)
    rerank = st.checkbox("Enable local reranking", value=False, disabled=strategy != "hybrid")
    question = st.text_area("Ask a question about indexed documents")
    if st.button("Ask", type="primary", disabled=not question.strip()):
        with st.spinner("Retrieving evidence and generating a local answer…"):
            try:
                result = requests.post(API + "/query", json={"question":question, "strategy":strategy, "top_k":top_k, "rerank":rerank}, timeout=240)
                result.raise_for_status(); data=result.json()
                st.markdown(data["answer"])
                st.caption(f"Retrieval: {data['latency_ms']['retrieval']:.0f} ms · Generation: {data['latency_ms']['generation']:.0f} ms")
                for citation in data["citations"]:
                    with st.expander(f"[{citation['marker']}] {citation['filename']}" + (f" · page {citation['page']}" if citation['page'] else "")):
                        st.write(citation["excerpt"])
                with st.expander("Retrieved passages and scores"):
                    for rank, chunk in enumerate(data["retrieved"], 1): st.write(f"{rank}. {chunk['filename']} · score {chunk['score']:.4f}\n\n{chunk['text']}")
                st.session_state.setdefault("history", []).append({"question":question,"answer":data["answer"]})
            except requests.RequestException as exc: st.error(str(exc))
    for item in reversed(st.session_state.get("history", [])[-5:]):
        with st.expander(item["question"]): st.markdown(item["answer"])

with tab_docs:
    files = st.file_uploader("Upload PDF, TXT, or Markdown", type=["pdf","txt","md","markdown"], accept_multiple_files=True)
    if st.button("Index selected files", disabled=not files):
        for file in files:
            try:
                response=requests.post(API+"/documents/upload", files={"files":(file.name,file.getvalue(),file.type)}, timeout=300); response.raise_for_status()
                st.json(response.json())
            except requests.RequestException as exc: st.error(f"{file.name}: {exc}")
    try:
        docs=requests.get(API+"/documents", timeout=5).json()
        for doc in docs:
            left,mid,right=st.columns([4,1,1]); left.write(f"**{doc['filename']}** · {doc['chunks']} chunks · {doc['status']}")
            if mid.button("Reindex", key="reindex_"+doc["document_id"]):
                try: st.json(requests.post(API+"/documents/"+doc["document_id"]+"/reindex", timeout=300).json())
                except requests.RequestException as exc: st.error(str(exc))
            if right.button("Delete", key="del_"+doc["document_id"]):
                requests.delete(API+"/documents/"+doc["document_id"], timeout=10).raise_for_status(); st.rerun()
    except requests.RequestException: st.info("Document inventory is unavailable until the API starts.")

with tab_eval:
    st.write("Retrieval metrics are deterministic and run without answer generation. Dense methods require the configured local Ollama embedding model; reranking is optional.")
    include_generation = st.checkbox("Also evaluate answers with local Ollama (slower)", value=False)
    if st.button("Run retrieval evaluation"):
        with st.spinner("Running benchmark…"):
            try: requests.post(API+"/eval/run", params={"generation":include_generation}, timeout=1800).raise_for_status(); st.success("Evaluation finished.")
            except requests.RequestException as exc: st.error(str(exc))
    try:
        report=requests.get(API+"/eval/results", timeout=10)
        if report.ok:
            data=report.json(); st.caption(f"Experiment {data['experiment_id']} · dataset {data['dataset_hash'][:12]} · corpus {data['corpus_hash'][:12]}")
            rows=[{"strategy":r["strategy"],"status":r["status"],**r.get("metrics",{})} for r in data["results"]]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            if os.path.exists("reports/latest.png"): st.image("reports/latest.png")
            elif os.path.exists("reports/latest.svg"):
                svg=base64.b64encode(open("reports/latest.svg","rb").read()).decode("ascii")
                st.markdown(f'<img alt="Retrieval metrics chart" src="data:image/svg+xml;base64,{svg}" width="780">', unsafe_allow_html=True)
            st.download_button("Download JSON", report.text, file_name="evaluation.json", mime="application/json")
            try:
                csv=requests.get(API+"/eval/results", timeout=10).text
                st.download_button("Download CSV", pd.DataFrame(rows).to_csv(index=False), file_name="evaluation.csv", mime="text/csv")
            except Exception: pass
    except requests.RequestException: st.info("No evaluation results are available yet.")
