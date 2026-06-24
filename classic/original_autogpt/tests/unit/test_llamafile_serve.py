import importlib.util
from pathlib import Path

import pytest

SERVE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "llamafile" / "serve.py"


def _load_serve_module():
    spec = importlib.util.spec_from_file_location("llamafile_serve", SERVE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


serve = _load_serve_module()


def test_default_llamafile_urls_are_allowed():
    serve._assert_download_url_allowed(serve.LLAMAFILE_URL)
    serve._assert_download_url_allowed(serve.LLAMAFILE_EXE_URL)


def test_allowed_huggingface_url_passes():
    serve._assert_download_url_allowed(
        "https://huggingface.co/some-org/some-model/resolve/main/model.llamafile"
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://huggingface.co/some-org/some-model/resolve/main/model.llamafile",
        "https://169.254.169.254/latest/meta-data/",
        "https://evil.example.com/model.llamafile",
        "https://huggingface.co.evil.example.com/model.llamafile",
    ],
)
def test_disallowed_urls_are_rejected(url: str):
    with pytest.raises(ValueError):
        serve._assert_download_url_allowed(url)
