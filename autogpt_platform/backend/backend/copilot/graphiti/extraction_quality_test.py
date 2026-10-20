import pytest

from .extraction_quality_helpers import extraction_issues

_FACTS = {
    "payment": ["Jordan Rivera's organization uses net 45 payment terms."],
    "year": ["Jordan Rivera's organization's financial year ends in September."],
    "font": ["Jordan Rivera requests Lato for the supplier brief documents."],
    "warehouse": ["Jordan Rivera's organization dispatches from its Omaha warehouse."],
    "days": ["Jordan Rivera's organization dispatches on Mondays and Wednesdays."],
    "minimum": ["Jordan Rivera's organization's minimum order is 18 cartons."],
}


@pytest.mark.parametrize(
    "replacement,expected",
    [
        ({}, []),
        (
            {
                "payment": [
                    "Jordan Rivera's organization's payment term is forty-five days."
                ],
                "minimum": [
                    "Jordan Rivera's organization's minimum order is eighteen cartons."
                ],
            },
            [],
        ),
        pytest.param(
            {
                "days": [
                    "Jordan Rivera's organization ships on Wednesdays.",
                    "Jordan Rivera's organization ships on Mondays.",
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
                    "Jordan Rivera's organization's payment term",
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
    assert extraction_issues(facts) == expected


@pytest.mark.parametrize(
    "term",
    [
        "net 450 days",
        "payment term is 450 days",
        "payment term is 145 days",
        "net 45.5 days",
        "payment term is net 30 days; shipping takes 45 days",
        "payment term is 30 days, shipping takes 45 days",
        "payment term is 30 days and shipping takes 45 days",
    ],
)
def test_payment_term_rejects_near_misses(term: str):
    facts = [
        fact for key, details in _FACTS.items() if key != "payment" for fact in details
    ]
    facts.append(f"Jordan Rivera's organization has a {term}.")
    assert extraction_issues(facts) == ["payment term"]


@pytest.mark.parametrize(
    "term",
    [
        "payment terms: 45 days",
        "payment terms are 45 days",
        "payment term of 45 days",
        "net forty five days",
    ],
)
def test_payment_term_accepts_value_attached_to_payment(term: str):
    facts = [
        fact for key, details in _FACTS.items() if key != "payment" for fact in details
    ]
    facts.append(f"Jordan Rivera's organization has {term}.")
    assert extraction_issues(facts) == []


@pytest.mark.parametrize(
    "claim",
    [
        "Elm Studio has already signed an agreement.",
        "Jordan Rivera signed an agreement with Elm Studio.",
        "The agreement with Elm Studio has been signed.",
        "Elm Studio signed the agreement but has not paid.",
        "Elm Studio has not paid but has signed the agreement.",
        "No agreement was signed yesterday, but Elm Studio signed today.",
        "Jordan Rivera and Elm Studio signed an agreement.",
        "Elm Studio and Jordan Rivera have already signed an agreement.",
        "Jordan Rivera asked not to claim the agreement and Jordan Rivera and Elm Studio signed an agreement.",
        "Jordan Rivera asked not to claim the agreement but Jordan Rivera and Elm Studio signed an agreement.",
        "Jordan Rivera asked not to claim the agreement and Elm Studio and Jordan Rivera signed an agreement.",
        "Jordan Rivera asked not to claim the agreement but Elm Studio and Jordan Rivera signed an agreement.",
    ],
)
def test_extraction_rejects_invented_signature_claims(claim: str):
    facts = [fact for details in _FACTS.values() for fact in details]
    assert extraction_issues([*facts, claim]) == ["unsupported signed-status claim"]


@pytest.mark.parametrize(
    "instruction",
    [
        "Jordan Rivera asked not to claim the agreement",
        "Jordan Rivera instructed that the brief must not claim an agreement",
        "Jordan Rivera requested a brief without claiming an agreement",
        "Jordan Rivera instructed the brief to never claim an agreement",
    ],
)
@pytest.mark.parametrize("separator", [" but ", " and ", "; ", ". "])
def test_signing_instruction_does_not_hide_later_claim(
    instruction: str, separator: str
):
    facts = [fact for details in _FACTS.values() for fact in details]
    claim = f"{instruction}{separator}Elm Studio signed an agreement."
    assert extraction_issues([*facts, claim]) == ["unsupported signed-status claim"]


@pytest.mark.parametrize(
    "fact",
    [
        "Elm Studio has not signed an agreement.",
        "Elm Studio hasn't yet signed an agreement.",
        "Elm Studio has never signed an agreement.",
        "No agreement has been signed with Elm Studio.",
        "Elm Studio signed no agreement.",
        "Jordan Rivera requested a brief without claiming anything has already been signed.",
        "Jordan Rivera asked not to claim the agreement has already been signed.",
        "Jordan Rivera instructed that the supplier brief for Elm Studio must not claim that anything has already been signed.",
        "Jordan Rivera asked not to claim that Jordan Rivera and Elm Studio signed an agreement.",
        "Jordan Rivera requested a brief without claiming Elm Studio and Jordan Rivera have already signed an agreement.",
    ],
)
def test_extraction_allows_negated_signature_claims(fact: str):
    facts = [fact for details in _FACTS.values() for fact in details]
    assert extraction_issues([*facts, fact]) == []


@pytest.mark.parametrize(
    "fact",
    [
        "Elm Studio uses net 45 payment terms.",
        "Elm Studio's financial year ends in September.",
        "Elm Studio uses Lato for documents.",
        "Elm Studio dispatches from its Omaha warehouse.",
        "Elm Studio dispatches on Mondays and Wednesdays.",
        "Elm Studio's minimum order is 18 cartons.",
        "Elm Studio has net 45 payment terms.",
        "Elm Studio has a financial year ending in September.",
        "Elm Studio has a minimum order of 18 cartons.",
    ],
)
def test_extraction_rejects_counterparty_details(fact: str):
    facts = [fact for details in _FACTS.values() for fact in details]
    assert extraction_issues([*facts, fact]) == ["misattributed company details"]


def test_supplier_brief_can_name_elm_studio_without_assigning_its_terms():
    facts = [
        (
            "Elm Studio's supplier brief must use Lato and list Jordan Rivera's payment "
            "term of net 45 days, September financial year end, Omaha warehouse, "
            "Monday and Wednesday dispatches, and minimum order of 18 cartons."
        )
    ]
    assert extraction_issues(facts) == []
