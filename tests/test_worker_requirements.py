from pathlib import Path
import re


def test_worker_requirements_include_onnx_for_insightface():
    requirements = Path("requirements-worker.txt").read_text(encoding="utf-8")

    assert re.search(r"^onnx(?:[<>=].*)?$", requirements, re.MULTILINE)