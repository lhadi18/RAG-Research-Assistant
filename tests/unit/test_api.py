from fastapi.testclient import TestClient
from rag_assistant.api.main import app


def test_api_schema_and_empty_documents():
    client=TestClient(app)
    assert client.get("/health").status_code==200
    assert client.get("/documents").status_code==200
    assert client.get("/openapi.json").status_code==200
    assert client.post("/query",json={"question":"x","strategy":"nope"}).status_code==422
    assert client.post("/documents/upload").status_code==422
