import pytest

from .extraction_quality_integration_test import _missing_details

_FACTS = {
    "payment": ["Elm Studio uses net 45 payment terms."],
    "year": ["Elm Studio's financial year ends in September."],
    "font": ["Elm Studio uses Lato for documents."],
    "warehouse": ["Elm Studio dispatches from its Omaha warehouse."],
    "days": ["Elm Studio dispatches on Mondays and Wednesdays."],
    "minimum": ["Elm Studio's minimum order is 18 cartons."],
}


@pytest.mark.parametrize(
    "replacement,expected",
    [
        ({}, []),
        (
            {
                "payment": ["Elm Studio's payment term is forty-five days."],
                "minimum": ["Elm Studio's minimum order is eighteen cartons."],
            },
            [],
        ),
        pytest.param(
            {
                "days": [
                    "Elm Studio ships on Wednesdays.",
                    "Elm Studio ships on Mondays.",
                ]
            },
            [],
        ),
        pytest.param(
            {"days": ["The staff meetings are on Monday and Wednesday."]},
            ["dispatch days"],
        ),
        pytest.param(
            {
                "payment": [
                    "Elm Studio's payment term",
                    "45 days remain until the audit.",
                ]
            },
            ["payment term"],
        ),
        pytest.param(
            {"minimum": ["The number of staff is 18", "cartons are blue."]},
            ["minimum order"],
        ),
    ],
)
def test_details_require_coherent_facts_without_row_order(replacement, expected):
    facts = [fact for detail in (_FACTS | replacement).values() for fact in detail]
    assert _missing_details(facts) == expected
