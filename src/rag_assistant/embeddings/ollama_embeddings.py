from rag_assistant.generation.ollama_client import OllamaClient


class OllamaEmbeddings:
    def __init__(self, client: OllamaClient):
        self.client = client

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.client.embed(texts)
