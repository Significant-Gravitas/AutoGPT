from backend.copilot.bot.sent_messages import _key


def test_key_keeps_colon_ids_apart():
    # Teams channel and message ids both contain ':'. Joined naively, these two
    # different messages would collapse onto the same authorship record.
    assert _key("teams", "a:b", "c") != _key("teams", "a", "b:c")


def test_key_is_stable_and_scoped_by_platform():
    assert _key("discord", "1", "2") == _key("discord", "1", "2")
    assert _key("discord", "1", "2") != _key("slack", "1", "2")
