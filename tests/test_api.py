import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from api.main import app


def test_answer_endpoint_returns_route_metadata():
    client = TestClient(app)
    response = client.post("/answer", json={"question": "How do I approve a purchase order?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["contexts"]
    assert payload["metadata"]["complexity_label"] in {"simple", "moderate", "complex"}

