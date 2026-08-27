"""How a subscription provider shows up in the connections list.

Two facts used to live in the client as literals keyed on ``codex``: that a
ChatGPT subscription files under the OpenAI card, and what that card then
says. Both are properties of the provider, so both moved here -- and these
are the assertions that keep the next provider from needing a client edit.
"""

from backend.api.features.integrations.models import (
    ProviderMetadata,
    display_alias_for,
    merge_subscription_summaries,
)


def test_a_subscription_files_under_the_card_a_person_looks_on() -> None:
    assert display_alias_for("codex") == "openai"


def test_a_subscription_that_would_collide_keeps_its_own_card() -> None:
    """A card shows one tab per auth method. GitHub already offers OAuth for
    repositories, so filing Copilot under it would put two different OAuth
    sign-ins on one tab and leave the loser unreachable -- the connection
    would exist, be entitled, and have nowhere to be clicked."""
    assert display_alias_for("github_copilot") is None


def test_an_ordinary_provider_is_its_own_entry() -> None:
    assert display_alias_for("github") is None
    assert display_alias_for("notion") is None


def test_the_card_says_both_things_it_can_be() -> None:
    merged = {
        row.name: row.description
        for row in merge_subscription_summaries(
            [
                ProviderMetadata(name="openai", description="GPT models"),
                ProviderMetadata(name="codex", description="internal detail"),
            ]
        )
    }
    assert merged["openai"] == "GPT models or your ChatGPT subscription"
    # The aliased row keeps its own description: it is not rendered as a card,
    # and rewriting it would hide which credential a caller actually holds.
    assert merged["codex"] == "internal detail"


def test_a_hidden_provider_does_not_advertise_itself() -> None:
    """The entitlement gate drops the row before this runs, and the card must
    follow -- offering "or your ChatGPT subscription" to someone who cannot
    connect one is an ad for a locked door."""
    merged = merge_subscription_summaries(
        [ProviderMetadata(name="openai", description="GPT models")]
    )
    assert merged[0].description == "GPT models"


def test_a_card_with_no_description_still_reads_as_a_sentence() -> None:
    merged = merge_subscription_summaries(
        [
            ProviderMetadata(name="openai", description=None),
            ProviderMetadata(name="codex"),
        ]
    )
    assert merged[0].description == "openai models or your ChatGPT subscription"
