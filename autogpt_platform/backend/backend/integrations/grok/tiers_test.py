"""The naming trap, pinned first because it is the whole reason this exists.

`GrokPro` is the $30 SuperGrok plan. `SuperGrokPro` is Heavy. The product
name that starts with "SuperGrok" is not the SuperGrok plan and the one that
does not, is -- so any string rule mis-buckets every plan, and mis-buckets
them into each other's price bands rather than into an error anyone would
notice.
"""

import base64
import json

from backend.integrations.grok.tiers import (
    FREE_TIER,
    UNKNOWN_TIER,
    is_paid,
    tier_for_product,
    tier_from_access_token,
)


def token_with(claim) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"tier": claim}).encode()).decode()
    return f"header.{payload.rstrip('=')}.signature"


class TestTheNamingTrap:
    def test_grokpro_is_supergrok_and_supergrokpro_is_heavy(self) -> None:
        assert tier_for_product("GrokPro") == "supergrok"
        assert tier_for_product("SuperGrokPro") == "supergrok_heavy"

    def test_no_string_rule_would_have_got_this_right(self) -> None:
        """Stated as an assertion so a future "simplification" to
        `startswith` fails here instead of in someone's billing."""
        assert not tier_for_product("GrokPro").startswith("supergrok_")
        assert tier_for_product("SuperGrokPro") != "supergrok"

    def test_every_product_name_maps_somewhere_explicit(self) -> None:
        for name, expected in (
            ("GrokPro", "supergrok"),
            ("SuperGrokPro", "supergrok_heavy"),
            ("SuperGrokPlus", "supergrok_plus"),
            ("SuperGrokLite", "supergrok_lite"),
            ("XBasic", "x_basic"),
            ("XPremium", "x_premium"),
            ("XPremiumPlus", "x_premium_plus"),
        ):
            assert tier_for_product(name) == expected

    def test_an_unrecognised_name_is_unknown_rather_than_guessed(self) -> None:
        """A new plan should read as "we do not know" -- guessing puts an
        account in the wrong band silently."""
        assert tier_for_product("SuperGrokUltra") == UNKNOWN_TIER
        assert tier_for_product("") == UNKNOWN_TIER
        assert tier_for_product(None) == UNKNOWN_TIER


class TestFreeIsAWorkingConnection:
    def test_free_is_recognised_rather_than_unknown(self) -> None:
        assert tier_for_product("Free") == FREE_TIER

    def test_is_paid_answers_about_headroom_not_about_access(self) -> None:
        """On xAI a plan raises a limit rather than unlocking access. A free
        account runs chats; treating it as unusable turns a supported state
        into an onboarding failure."""
        assert is_paid("supergrok") is True
        assert is_paid("x_basic") is True
        assert is_paid(FREE_TIER) is False
        assert is_paid(UNKNOWN_TIER) is False
        assert is_paid(None) is False

    def test_all_seven_paid_products_count_as_paid(self) -> None:
        for name in (
            "GrokPro",
            "SuperGrokPro",
            "SuperGrokPlus",
            "SuperGrokLite",
            "XBasic",
            "XPremium",
            "XPremiumPlus",
        ):
            assert is_paid(tier_for_product(name)) is True


class TestReadingTheTokenClaim:
    def test_reads_the_ordinal_form(self) -> None:
        assert tier_from_access_token(token_with(0)) == FREE_TIER
        assert tier_from_access_token(token_with(1)) == "supergrok"
        assert tier_from_access_token(token_with(5)) == "supergrok_heavy"
        assert tier_from_access_token(token_with(7)) == "supergrok_plus"

    def test_reads_the_name_form(self) -> None:
        assert tier_from_access_token(token_with("supergrok_heavy")) == (
            "supergrok_heavy"
        )

    def test_an_unmapped_ordinal_is_reported_as_itself(self) -> None:
        """A tier we have not seen is still worth carrying verbatim -- it
        ends up in a log where someone can recognise it."""
        assert tier_from_access_token(token_with(99)) == "99"

    def test_a_boolean_claim_does_not_read_as_tier_one(self) -> None:
        """`bool` is an `int` in Python, so `True` would quietly map to
        "supergrok" and hand a free account a paid label."""
        assert tier_from_access_token(token_with(True)) == UNKNOWN_TIER

    def test_a_token_it_cannot_read_is_unknown_rather_than_an_error(self) -> None:
        """A malformed claim is not a reason to fail a chat the server may
        well accept."""
        assert tier_from_access_token("not-a-jwt") == UNKNOWN_TIER
        assert tier_from_access_token("only.two") == UNKNOWN_TIER
        assert tier_from_access_token("a.!!!not-base64!!!.c") == UNKNOWN_TIER
        assert tier_from_access_token("") == UNKNOWN_TIER

    def test_unpadded_base64url_decodes(self) -> None:
        """JWT segments drop the padding that `b64decode` insists on --
        forgetting to re-add it makes every real token unreadable."""
        assert tier_from_access_token(token_with("x_premium")) == "x_premium"
