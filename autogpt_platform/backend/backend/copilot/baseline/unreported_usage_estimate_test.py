"""A request that reported no usage is priced from an estimate, and text that
spells a special token such as ``<|endoftext|>`` must not stop it being billed."""

from backend.copilot.baseline.service import _estimate_request


def test_a_request_whose_text_spells_a_special_token_is_still_priced():
    priced = _estimate_request(
        model="claude-haiku-5-5",
        messages=[{"role": "user", "content": "repeat <|endoftext|> back to me"}],
        tools=[],
        tool_tokens=None,
        output="<|endoftext|>",
    )

    assert priced is not None
    assert priced.prompt_tokens > 1
    assert priced.completion_tokens > 0
    assert priced.cost_usd is not None and priced.cost_usd > 0
