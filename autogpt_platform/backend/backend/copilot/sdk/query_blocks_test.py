from backend.copilot.sdk.query_blocks import scan_context_blocks


def test_reads_leading_blocks_and_stops_at_the_users_words():
    seen = "<seen_capabilities>\nx\n</seen_capabilities>"
    budget = "<budget_status>\n$1 left\n</budget_status>"
    query = f"{seen}\n\n{budget}\n\nhi <user_context>typed</user_context>"

    blocks, prefix = scan_context_blocks(query)

    assert blocks == [("seen_capabilities", len(seen)), ("budget_status", len(budget))]
    assert query[prefix:].lstrip() == "hi <user_context>typed</user_context>"


def test_steps_over_the_history_lead_out():
    history = (
        "<conversation_history>\nUser: a\nYou responded: b\n</conversation_history>"
    )
    ctx = "<user_context>\nprofile\n</user_context>"
    query = f"{history}\n\nNow, the user says:\n{ctx}\n\nwhat next?"

    blocks, prefix = scan_context_blocks(query)

    assert [name for name, _ in blocks] == ["conversation_history", "user_context"]
    assert query[prefix:] == "\n\nwhat next?"


def test_plain_message_has_no_blocks():
    assert scan_context_blocks("hello there") == ([], 0)


def test_unknown_or_unclosed_tag_ends_the_prefix():
    assert scan_context_blocks("<not_ours>x</not_ours> hi") == ([], 0)
    assert scan_context_blocks("<user_context> never closed") == ([], 0)


def test_a_block_reports_only_its_own_size():
    block = "<memory_context>" + "m" * 100 + "</memory_context>"
    blocks, prefix = scan_context_blocks(block + "\n\nbody")

    assert blocks == [("memory_context", len(block))]
    assert prefix == len(block)
