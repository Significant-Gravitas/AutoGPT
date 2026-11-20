"""The gist rules are what stop raw output landing in an inbox."""

from backend.notifications.gist import GIST_MAX_CHARS, build_gist, fallback_gist


def test_activity_status_wins_and_is_one_sentence():
    gist = build_gist(
        {},
        "Matched 41 new leads to the saved search. It also refreshed the cache.",
    )
    assert gist == "Matched 41 new leads to the saved search."


def test_structured_output_is_counted_not_pasted():
    leads = [{"name": f"lead {i}"} for i in range(41)]
    gist = build_gist({"leads": leads}, None)
    assert gist is not None
    assert "41 leads" in gist
    assert "lead 0" not in gist


def test_file_outputs_are_named_never_embedded():
    gist = build_gist(
        {"clips": ["data:video/mp4;base64,AAAA", "data:video/mp4;base64,BBBB"]},
        None,
    )
    assert gist == "produced 2 clips."
    assert "base64" not in gist


def test_long_text_is_described_not_reproduced():
    draft = " ".join(["word"] * 1200)
    gist = build_gist({"draft": [draft]}, None)
    assert gist is not None
    assert gist == "wrote a 1,200-word result."
    assert len(gist) <= GIST_MAX_CHARS + 1


def test_gist_is_length_capped():
    gist = build_gist({}, "x" * 500)
    assert gist is not None
    assert len(gist) <= GIST_MAX_CHARS + 1
    assert gist.endswith("…")


def test_no_output_and_no_summary_has_nothing_to_say():
    assert build_gist({}, None) is None
    assert build_gist({"result": [""]}, "   ") is None


def test_final_fallback_only_claims_what_we_know():
    assert fallback_gist(1) == "completed 1 run."
    assert fallback_gist(14) == "completed 14 runs."
