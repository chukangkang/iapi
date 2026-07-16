import base64
import io

from fastapi.testclient import TestClient
from PIL import Image

from supir_worker.api import create_app
from supir_worker.settings import SupirWorkerSettings


def _encoded_image() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (16, 12), "red").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_health_reports_backend_readiness():
    client = TestClient(create_app(SupirWorkerSettings(_env_file=None), FakeBackend()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["model"] == "Q"


def test_restore_requires_configured_bearer_token():
    settings = SupirWorkerSettings(_env_file=None, api_key="secret")
    client = TestClient(create_app(settings, FakeBackend()))

    response = client.post("/v1/restore", json={"image": _encoded_image(), "width": 64, "height": 64})

    assert response.status_code == 401


def test_restore_returns_base64_png_and_timing():
    backend = FakeBackend()
    settings = SupirWorkerSettings(_env_file=None, api_key="secret")
    client = TestClient(create_app(settings, backend))

    response = client.post(
        "/v1/restore",
        headers={"Authorization": "Bearer secret"},
        json={"image": _encoded_image(), "prompt": "natural photo", "width": 64, "height": 64},
    )

    assert response.status_code == 200
    payload = response.json()
    restored = Image.open(io.BytesIO(base64.b64decode(payload["image"])))
    assert restored.size == (64, 64)
    assert payload["model"] == "Q"
    assert payload["elapsed_ms"] >= 0
    assert backend.last_prompt == "natural photo"


def test_restore_rejects_invalid_base64():
    client = TestClient(create_app(SupirWorkerSettings(_env_file=None), FakeBackend()))

    response = client.post("/v1/restore", json={"image": "not-base64", "width": 64, "height": 64})

    assert response.status_code == 400


class FakeBackend:
    ready = True
    model_sign = "Q"
    last_prompt = None

    def restore(self, image, *, prompt, width, height):
        self.last_prompt = prompt
        return image.resize((width, height))