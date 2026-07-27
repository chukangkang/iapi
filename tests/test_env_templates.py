import re
from pathlib import Path

import pytest

from app.config import Settings
from supir_worker.settings import SupirWorkerSettings


APP_ENV_TEMPLATES = (
    Path(".env.example"),
    Path(".env.api.example"),
    Path(".env.worker.example"),
)
SUPIR_ENV_TEMPLATE = Path("deploy/supir/.env.example")
ENV_KEY_PATTERN = re.compile(r"^(?:#\s*)?([A-Z][A-Z0-9_]*)=", re.MULTILINE)


def _template_keys(path: Path) -> list[str]:
    return ENV_KEY_PATTERN.findall(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", APP_ENV_TEMPLATES)
def test_app_env_templates_can_be_loaded(path: Path, monkeypatch):
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)

    settings = Settings(_env_file=path)

    assert settings.oss_enabled is False


def test_full_app_env_templates_cover_every_setting_once():
    expected = {name.upper() for name in Settings.model_fields}

    for path in (Path(".env.example"), Path(".env.worker.example")):
        keys = _template_keys(path)
        assert set(keys) == expected
        assert len(keys) == len(set(keys))


def test_api_env_template_only_contains_supported_settings():
    expected = {name.upper() for name in Settings.model_fields}
    keys = _template_keys(Path(".env.api.example"))

    assert set(keys) <= expected
    assert len(keys) == len(set(keys))


def test_supir_env_template_can_be_loaded_and_covers_settings(monkeypatch):
    for name in SupirWorkerSettings.model_fields:
        monkeypatch.delenv(f"SUPIR_WORKER_{name.upper()}", raising=False)
    monkeypatch.chdir(SUPIR_ENV_TEMPLATE.parent)
    SupirWorkerSettings(_env_file=".env.example")

    keys = _template_keys(Path(".env.example"))
    expected = {f"SUPIR_WORKER_{name.upper()}" for name in SupirWorkerSettings.model_fields}
    assert set(keys) == expected | {"SUPIR_MODELS_DIR"}
    assert len(keys) == len(set(keys))


def test_compose_env_references_are_declared_in_supir_template():
    compose = Path("deploy/supir/compose.yaml").read_text(encoding="utf-8")
    referenced = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*)", compose))

    assert referenced <= set(_template_keys(SUPIR_ENV_TEMPLATE))
