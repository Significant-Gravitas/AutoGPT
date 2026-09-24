from backend.api.features.experts.avatar_catalog import resolve_avatar_url


def test_existing_roster_avatar_maps_without_using_the_name():
    assert (
        resolve_avatar_url("/experts/maria.svg")
        == "/experts/clay/v5/maria-marketing.png"
    )


def test_uploaded_and_saved_generated_avatars_are_preserved():
    for url in (
        "https://cdn.example/avatar.png",
        "/api/store/media/user/images/custom.png",
        "/avatars/mine.svg",
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


def test_each_builtin_has_its_own_avatar():
    from backend.api.features.experts.seed import ROSTER

    urls = [resolve_avatar_url(entry["avatar_url"]) for entry in ROSTER]
    assert len(urls) == len(set(urls))


def test_builtin_assets_are_distinct_and_palette_matches_request_schema():
    import hashlib
    from pathlib import Path
    from typing import get_args

    from PIL import Image

    from backend.api.features.experts.avatar_catalog import CATALOG, AvatarColor

    public = Path(__file__).parent.parents[4] / "frontend/public"
    assert {c.id for c in CATALOG.colors} == set(get_args(AvatarColor))
    assert len(CATALOG.identities) == 32
    hashes = set()
    for avatar in CATALOG.identities:
        path = public / avatar.url.lstrip("/")
        hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
        assert avatar.color_id in get_args(AvatarColor)
        with Image.open(path) as image:
            if avatar.url.endswith(".png"):
                assert image.mode == "RGBA"
                assert image.getchannel("A").getextrema() == (0, 255)
                assert image.size == (512, 512)
    assert len(hashes) == len(CATALOG.identities)


def test_accent_refresh_handles_each_known_default_and_preserves_other_choices():
    from backend.api.features.experts.avatar_catalog import (
        CATALOG,
        resolve_builtin_avatar_url,
    )

    for avatar in CATALOG.identities:
        for old in [avatar.previous_url, *avatar.previous_urls]:
            assert resolve_builtin_avatar_url(avatar.name, old) == avatar.url
        for custom in [
            "https://cdn.test/custom.png",
            "/experts/clay/v1/development.png",
        ]:
            if custom != avatar.previous_url:
                assert resolve_builtin_avatar_url(avatar.name, custom) == custom


def test_every_builtin_category_has_distinct_transparent_artwork():
    from pathlib import Path

    from PIL import Image

    from backend.api.features.experts.avatar_catalog import CATALOG, PRESETS
    from backend.api.features.experts.seed import ROSTER

    public = Path(__file__).parent.parents[4] / "frontend/public"
    by_name = {avatar.name: avatar for avatar in CATALOG.identities}
    for entry in ROSTER:
        avatar = by_name[entry["name"]]
        assert avatar.primary_category == entry["categories"][0]
        assert set(avatar.variants) == set(entry["categories"])
        assert avatar.color_id == PRESETS[avatar.primary_category].color_id
        assert avatar.url == avatar.variants[avatar.primary_category].url
        assert len({v.url for v in avatar.variants.values()}) == len(avatar.variants)
        for variant in avatar.variants.values():
            with Image.open(public / variant.url.lstrip("/")) as image:
                if variant.url.endswith(".png"):
                    assert image.size == (512, 512)
                    assert image.mode == "RGBA"
                    assert image.getchannel("A").getextrema() == (0, 255)


def test_a_picker_preset_doubles_as_an_old_default_so_hires_must_keep_it():
    """Five presets are also some expert's `previous_url`, so a hire sitting
    on one cannot be told apart from a hire whose owner chose it. `seed.py`
    skips the migration for these; this records why that guard is there."""
    from backend.api.features.experts.avatar_catalog import (
        CATALOG,
        PRESET_AVATAR_URLS,
        resolve_builtin_avatar_url,
    )

    assert PRESET_AVATAR_URLS == {avatar.url for avatar in CATALOG.avatars}
    shared = [
        (avatar.name, url)
        for avatar in CATALOG.identities
        for url in [avatar.previous_url, *avatar.previous_urls]
        if url in PRESET_AVATAR_URLS
    ]
    assert shared, "no overlap left — seed's PRESET_AVATAR_URLS guard can go"
    for name, url in shared:
        # Unguarded, the backfill rewrites a preset its owner picked.
        assert resolve_builtin_avatar_url(name, url) != url
