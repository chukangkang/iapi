import asyncio

from PIL import Image

from app.config import Settings
from app.restoration.supir_client import SupirClient


def test_supir_client_is_disabled_without_url():
    client = SupirClient(Settings(_env_file=None, supir_enabled=True, supir_base_url=""))

    assert client.available is False


def test_supir_failure_falls_back_to_input(monkeypatch):
    client = SupirClient(
        Settings(
            _env_file=None,
            supir_enabled=True,
            supir_base_url="http://supir:8000",
        )
    )
    image = Image.new("RGB", (32, 32), "red")

    monkeypatch.setattr(client, "_restore_sync", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline")))

    restored = asyncio.run(client.restore(image, prompt="restore", width=64, height=64))

    assert restored is image
