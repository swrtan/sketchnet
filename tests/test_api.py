from fastapi.testclient import TestClient

from sketchnet import server
from sketchnet.model import SketchCNN


def client():
    return TestClient(server.app)


def test_status_exposes_serialized_classes_and_missing_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "CHECKPOINT_PATH", tmp_path / "missing.pt")
    with client() as c:
        server.load_model()
        response = c.get("/api/status")
    assert response.status_code == 200
    body = response.json()
    assert body["ready"] is False
    assert isinstance(body["classes"], list)


def test_empty_request_and_validation():
    with client() as c:
        empty = c.post("/api/predict", json={"strokes": []})
        malformed = c.post("/api/predict", json={"strokes": [[[1, 2]]]})
    assert empty.status_code == 200 and empty.json()["empty"] is True
    assert malformed.status_code == 422


def test_predict_deterministic_order_and_probability_sum(monkeypatch):
    candidate = SketchCNN().eval()
    with client() as c:
        monkeypatch.setattr(server, "model", candidate)
        payload = {"strokes": [[[20, 80], [60, 20]], [[60, 20], [100, 80]]]}
        first, second = c.post("/api/predict", json=payload), c.post("/api/predict", json=payload)
    assert first.status_code == second.status_code == 200
    a, b = first.json(), second.json()
    assert abs(sum(a["probabilities"]) - 1) < 1e-5
    assert [x["label"] for x in a["top"]] == [x["label"] for x in b["top"]]
    assert all(x["probability"] >= y["probability"] for x, y in zip(a["top"], a["top"][1:]))


def test_inference_error_returns_500(monkeypatch):
    class Broken:
        def features(self, _):
            raise RuntimeError("boom")

    with client() as c:
        monkeypatch.setattr(server, "model", Broken())
        response = c.post("/api/predict", json={"strokes": [[[20, 80], [60, 20]]]})
    assert response.status_code == 500


def test_malformed_json_is_rejected():
    with client() as c:
        response = c.post("/api/predict", content=b"{not json", headers={"content-type": "application/json"})
    assert response.status_code == 422
