from .models import CapabilityEntry, Connection, clip_purpose, normalize_text


def test_clip_purpose_keeps_short_text_verbatim():
    assert clip_purpose("  Send an   email. ") == "Send an email."


def test_clip_purpose_cuts_at_sentence_boundary():
    text = "Create an issue in Linear with a title and team. " + "x" * 200
    assert clip_purpose(text) == "Create an issue in Linear with a title and team."


def test_clip_purpose_falls_back_to_word_boundary_with_ellipsis():
    text = "word " * 60
    clipped = clip_purpose(text)
    assert clipped.endswith("…") and len(clipped) <= 160


def test_normalize_text_collapses_whitespace_and_bounds_the_index_text():
    assert normalize_text("  Send   an\nemail. ") == "Send an email."
    text = "word " * 3000
    bounded = normalize_text(text)
    assert len(bounded) <= 4000 and bounded.endswith("word")
    assert normalize_text("alpha beta gamma", limit=10) == "alpha beta"


def test_listing_is_compact_and_only_shows_connection_when_required():
    entry = CapabilityEntry(
        id="tool:web_search",
        kind="tool",
        name="web_search",
        purpose="Search.",
        description="Search. Returns the top results for a query.",
    )
    assert entry.listing() == {
        "id": "tool:web_search",
        "name": "web_search",
        "purpose": "Search.",
        "kind": "tool",
    }
    entry = CapabilityEntry(
        id="block:1",
        kind="block",
        klass="primitive",
        name="SendWebRequestBlock",
        purpose="HTTP.",
        connection=Connection(required=True, key_type="host"),
        eager=False,
    )
    listing = entry.listing()
    assert listing["class"] == "primitive" and listing["connected"] is None


def test_available_in_context():
    graph_only = CapabilityEntry(
        id="block:2", kind="block", name="AgentInputBlock", purpose="", context="graph"
    )
    assert graph_only.available_in("graph")
    assert not graph_only.available_in("direct")
    assert graph_only.available_in("both")
