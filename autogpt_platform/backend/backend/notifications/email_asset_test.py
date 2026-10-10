"""Email artwork stays transparent when clients recolor the surrounding band."""

from pathlib import Path

import pytest
from PIL import Image

from backend.notifications.template_env import create_body_environment

PUBLIC = Path(__file__).resolve().parents[3] / "frontend" / "public"
OTTO_PATH = "/autogpt-characters/v1.1/otto/neutral-transparent"


@pytest.mark.parametrize("size", [256, 512, 1024])
def test_email_otto_exports_have_transparent_backgrounds(size):
    path = PUBLIC / f"{OTTO_PATH.lstrip('/')}/{size}.png"
    assert path.is_file()
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.size == (size, size)
        assert "A" in image.getbands()
        alpha = image.getchannel("A")
        assert alpha.getextrema() == (0, 255)
        assert all(
            alpha.getpixel(point) == 0
            for point in ((0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1))
        )


def test_email_hero_default_uses_the_transparent_export():
    environment = create_body_environment(autoescape=True)
    html = environment.from_string(
        '{% import "_email_ui.j2" as ui with context %}{{ ui.hero_otto() }}'
    ).render()
    assert f'src="https://platform.agpt.co{OTTO_PATH}/256.png"' in html
    assert 'width="160" height="160"' in html
