import os

import pytest

os.environ["ARAGBIZ_USE_TRAINED_CLASSIFIER"] = "false"

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from api.main import app


def test_answer_endpoint_returns_route_metadata():
    client = TestClient(app)
    response = client.post("/answer", json={"question": "Can I start accepting payments while Wix Payments is under verification?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["contexts"]
    assert payload["metadata"]["complexity_label"] in {"simple", "moderate", "complex"}
