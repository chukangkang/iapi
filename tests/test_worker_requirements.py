from pathlib import Path
import re

import pytest


def test_worker_requirements_include_onnx_for_insightface():
    requirements = Path("requirements-worker.txt").read_text(encoding="utf-8")

    assert re.search(r"^onnx(?:[<>=].*)?$", requirements, re.MULTILINE)


@pytest.mark.parametrize("requirements_file", ["requirements-api.txt", "requirements.txt"])
def test_oss_requirements_include_pycryptodome(requirements_file: str):
    requirements = Path(requirements_file).read_text(encoding="utf-8")

    assert re.search(r"^pycryptodome(?:[<>=].*)?$", requirements, re.MULTILINE)


def test_worker_inherits_api_runtime_dependencies():
    requirements = Path("requirements-worker.txt").read_text(encoding="utf-8")

    assert re.search(r"^-r\s+requirements-api\.txt$", requirements, re.MULTILINE)