import httpx
import pytest
from rag_assistant.config import Settings
from rag_assistant.generation.ollama_client import OllamaClient, OllamaUnavailable, OllamaModelMissing, OllamaError


def test_hosted_provider_rejected():
    with pytest.raises(ValueError): Settings(llm_provider="openai")


def test_ollama_missing_model_and_timeout():
    def missing(request): return httpx.Response(404, request=request)
    client=OllamaClient(Settings(),httpx.Client(transport=httpx.MockTransport(missing),base_url="http://local"))
    with pytest.raises(OllamaModelMissing): client.embed(["test"])
    def timeout(request): raise httpx.ReadTimeout("timeout",request=request)
    client=OllamaClient(Settings(),httpx.Client(transport=httpx.MockTransport(timeout),base_url="http://local"))
    with pytest.raises(OllamaUnavailable): client.chat([])


def test_ollama_malformed_embedding_response():
    def malformed(request): return httpx.Response(200,json={"unexpected":[]},request=request)
    client=OllamaClient(Settings(),httpx.Client(transport=httpx.MockTransport(malformed),base_url="http://local"))
    with pytest.raises(OllamaError,match="malformed embeddings"): client.embed(["test"])
