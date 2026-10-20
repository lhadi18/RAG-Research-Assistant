from __future__ import annotations
import httpx
from rag_assistant.config import Settings, get_settings


class OllamaError(RuntimeError):
    pass


class OllamaUnavailable(OllamaError):
    pass


class OllamaModelMissing(OllamaError):
    pass


class OllamaClient:
    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None):
        self.settings = settings or get_settings()
        self.http = client or httpx.Client(base_url=self.settings.ollama_base_url.rstrip("/"),
                                           timeout=self.settings.ollama_timeout_seconds)

    def _request(self, method: str, path: str, **kwargs):
        try:
            response = self.http.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise OllamaUnavailable("Ollama request timed out. Check the service and timeout setting.") from exc
        except httpx.HTTPError as exc:
            raise OllamaUnavailable("Cannot reach Ollama. Start it locally and check OLLAMA_BASE_URL.") from exc
        if response.status_code == 404:
            raise OllamaModelMissing("Configured Ollama model is missing. Pull it with `ollama pull MODEL`.")
        if response.is_error:
            raise OllamaError(f"Ollama returned HTTP {response.status_code}.")
        try:
            return response.json()
        except ValueError as exc:
            raise OllamaError("Ollama returned malformed JSON.") from exc

    def status(self) -> dict:
        try:
            tags = self._request("GET", "/api/tags")
            names = {m.get("name") for m in tags.get("models", [])}
            chat = self.settings.ollama_chat_model in names
            embed = self.settings.ollama_embed_model in names
            return {"reachable": True, "chat_model": chat, "embedding_model": embed,
                    "missing": [m for m, ok in ((self.settings.ollama_chat_model, chat), (self.settings.ollama_embed_model, embed)) if not ok]}
        except OllamaError as exc:
            return {"reachable": False, "chat_model": False, "embedding_model": False, "detail": str(exc)}

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        data = self._request("POST", "/api/embed", json={"model": self.settings.ollama_embed_model, "input": texts})
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts) or any(not isinstance(v, list) or not v for v in vectors):
            raise OllamaError("Ollama returned malformed embeddings.")
        dims = {len(v) for v in vectors}
        if len(dims) != 1:
            raise OllamaError("Ollama returned inconsistent embedding dimensions.")
        return vectors

    def chat(self, messages: list[dict], stream: bool = False) -> str:
        data = self._request("POST", "/api/chat", json={"model": self.settings.ollama_chat_model,
            "messages": messages, "stream": False, "options": {"temperature": self.settings.ollama_temperature,
            "num_ctx": self.settings.ollama_num_ctx, "num_predict": self.settings.ollama_num_predict}})
        content = data.get("message", {}).get("content")
        if not isinstance(content, str):
            raise OllamaError("Ollama returned a malformed chat response.")
        return content
