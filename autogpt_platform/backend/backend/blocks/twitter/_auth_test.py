from backend.blocks import get_blocks
from backend.integrations.oauth.twitter import TwitterOAuthHandler
from backend.integrations.providers import ProviderName


def test_every_x_block_requires_only_scopes_the_login_asks_x_for():
    # The picker offers only a credential holding every required scope, and
    # get_login_url asks X for DEFAULT_SCOPES whatever the block requested.
    required = _required_scopes_by_x_block()
    assert required, "found no X blocks, so this test checks nothing"

    asked = set(TwitterOAuthHandler.DEFAULT_SCOPES)
    outside = {
        name: sorted(scopes - asked)
        for name, scopes in required.items()
        if not scopes <= asked
    }
    assert not outside, f"X blocks requiring scopes we never ask X for: {outside}"


def _required_scopes_by_x_block() -> dict[str, set[str]]:
    required: dict[str, set[str]] = {}
    for block_cls in get_blocks().values():
        block = block_cls()
        for info in block.input_schema.get_credentials_fields_info().values():
            if ProviderName.TWITTER in info.provider:
                required.setdefault(block.name, set()).update(
                    info.required_scopes or ()
                )
    return required
