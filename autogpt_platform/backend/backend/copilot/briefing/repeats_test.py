import pytest

from backend.copilot.briefing.repeats import TakenItem, is_repeat, stems

COMPARISON = "Compare dairy supplier offers and cost impact of paneer prices"


@pytest.mark.parametrize(
    "draft, taken",
    [
        # The eval's reworded repeats of a board task and a dismissed one.
        ("Work out the paneer price impact across dairy supplier offers", COMPARISON),
        (
            "Get a revised paneer rate from Kisan Fresh",
            "Chase Kisan Fresh for a revised paneer rate",
        ),
        (
            "Recap email for the Finlo budget call",
            "Write the Finlo budget call recap email",
        ),
        ("Pricing deck", "pricing DECK."),
    ],
)
def test_a_reword_is_a_repeat(draft, taken):
    assert is_repeat(draft, None, TakenItem(title=taken))


@pytest.mark.parametrize(
    "draft, taken",
    [
        ("Turn the dairy supplier comparison into a deck", COMPARISON),
        ("Ask Aarey Pure Dairy for a December paneer quote", COMPARISON),
        (
            "Set up search campaigns for Finlo's scenario B",
            "Finlo ROAS recovery plan with scenarios for the 9 Nov budget call",
        ),
        ("Book a pest control visit for Bandra", "Build the FSSAI inspection binder"),
        ("Email paneer prices to the chefs", "Paneer prices"),
    ],
)
def test_a_real_follow_up_is_not_a_repeat(draft, taken):
    assert not is_repeat(draft, None, TakenItem(title=taken))


def test_goals_count_when_both_sides_have_one():
    taken = TakenItem(
        title="Supplier review for December",
        goal="A December paneer cost per outlet from the Kisan Fresh offer",
    )
    draft_goal = "December paneer cost per outlet using the Kisan Fresh offer"
    assert is_repeat("December dairy costing", draft_goal, taken)
    assert not is_repeat("December dairy costing", None, taken)


def test_stems_fold_rewording_endings_and_drop_filler():
    assert stems("Get the revised prices") == stems("revise price")
    assert stems("Work out the next task") == set()
