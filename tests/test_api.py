"""API tests.

The detector is faked and injected into the app state so these tests run
without torch / onnxruntime / model weights. The lifespan loader (which would
load a real model) is bypassed by not using TestClient as a context manager.
"""
import io

import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src import api  # noqa: E402
from src.onnx_detector import Detection  # noqa: E402


class _FakeDetector:
    def predict(self, image):
        return [
            Detection(box=(10.0, 20.0, 110.0, 220.0), score=0.91,
                      class_id=0, class_name="person"),
            Detection(box=(300.0, 150.0, 420.0, 260.0), score=0.77,
                      class_id=2, class_name="car"),
        ]


@pytest.fixture()
def client():
    api.state["detector"] = _FakeDetector()
    api.state["backend"] = "fake"
    c = TestClient(api.app)          # no context manager -> lifespan not triggered
    yield c
    api.state.clear()


def _image_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (640, 480), color=(120, 120, 120)).save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_detect_returns_valid_schema(client):
    resp = client.post(
        "/detect",
        files={"file": ("test.jpg", _image_bytes(), "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2
    assert body["count"] == len(body["detections"])
    assert body["image_width"] == 640 and body["image_height"] == 480
    first = body["detections"][0]
    assert set(first) == {"box", "score", "class_id", "class_name"}
    assert len(first["box"]) == 4


def test_detect_rejects_non_image(client):
    resp = client.post(
        "/detect",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert resp.status_code == 400
