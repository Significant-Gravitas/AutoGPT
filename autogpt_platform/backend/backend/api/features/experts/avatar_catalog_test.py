from backend.api.features.experts.avatar_catalog import resolve_avatar_url


def test_existing_roster_avatar_maps_without_using_the_name():
    assert (
        resolve_avatar_url("/experts/maria.svg")
        == "/autogpt-characters/v1.1/expert-maria/neutral/128.webp"
    )


def test_uploaded_and_saved_generated_avatars_are_preserved():
    for url in (
        "https://cdn.example/avatar.png",
        "/api/store/media/user/images/custom.png",
        "/experts/clay/v1/finance.png",
    ):
        assert resolve_avatar_url(url) == url


def test_custom_legacy_avatar_gets_a_stable_non_otto_default():
    assert (
        resolve_avatar_url("/avatars/notion/1-2-3.violet.svg")
        == "/experts/clay/v1/content.png"
    )
    assert resolve_avatar_url(None) is None


def test_frontend_catalog_and_png_assets_match_backend():
    import json
    from pathlib import Path

    from PIL import Image

    folder = Path(__file__).parent
    frontend = folder.parents[4] / "frontend"
    catalog = json.loads((folder / "avatar_catalog.json").read_text())
    assert catalog == json.loads(
        (frontend / "src/components/molecules/ExpertAvatar/catalog.json").read_text()
    )
    assert len(catalog["legacy"]) == 32
    for url in [a["url"] for a in catalog["avatars"]]:
        with Image.open(frontend / "public" / url.lstrip("/")) as image:
            assert image.format == "PNG"
            assert image.size == (512, 512)
            assert image.mode == "RGBA"
            assert image.getchannel("A").getextrema() == (0, 255)
