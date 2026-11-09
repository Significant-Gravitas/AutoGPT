"""Scenario-specific assertions shared by offline and paid extraction tests."""

import re


def extraction_issues(facts: list[str]) -> list[str]:
    """Check concrete details and explicit unsupported claim forms in this scenario."""
    detail_name = r"(?:payment|financial|document|warehouse|dispatch|minimum)\b"
    # Reject explicit assignments of Jordan's details to the counterparty.
    counterparty_details = (
        rf"^\s*Elm Studio(?:['’]s\s+(?:normal\s+)?{detail_name}"
        r"|\s+(?:uses|dispatches|ships)\b"
        rf"|\s+has\s+(?:net\b|(?:(?:a|an|normal)\s+)?{detail_name}))"
    )
    violations = {
        "unsupported signed-status claim": any(
            _has_signed_claim(fact) for fact in facts
        ),
        "misattributed company details": any(
            re.search(counterparty_details, fact, re.IGNORECASE) for fact in facts
        ),
    }
    return _missing_details(facts) + [
        name for name, found in violations.items() if found
    ]


def _missing_details(facts: list[str]) -> list[str]:
    # Each dispatch day must occur in a fact about shipping or dispatch.
    dispatch = r"(?=.*\b(?:dispatch|ship|leav)\w*\b).*"
    # Accept exactly 45, including words, directly after a payment prefix.
    payment_days = r"(?:\b45\b(?![.,]\d)|\bforty[- ]five\b)"
    payment_prefix = r"\b(?:net\s+|payment(?:\s+terms?)?\s*(?:(?:is|are|of)\s+|:\s*)?)"
    expected = {
        "payment term": (payment_prefix + payment_days,),
        "financial year end": (r"September",),
        "document font": (r"Lato",),
        "warehouse": (r"Omaha",),
        "dispatch days": (dispatch + r"\bMondays?\b", dispatch + r"\bWednesdays?\b"),
        "minimum order": (r"(?:18|eighteen)\s+cartons",),
    }
    return [
        name
        for name, patterns in expected.items()
        if not all(
            any(re.search(pattern, fact, re.IGNORECASE) for fact in facts)
            for pattern in patterns
        )
    ]


def _has_signed_claim(fact: str) -> bool:
    # The scenario's two participants can form one subject joined by "and".
    coordinated_subject = (
        r"\b(?:Jordan Rivera\s+and\s+Elm Studio|" r"Elm Studio\s+and\s+Jordan Rivera)\b"
    )
    fact = re.sub(coordinated_subject, "the parties", fact, flags=re.IGNORECASE)
    # A negated instruction must not hide a later claim after "but" or "and".
    negated = (
        r"\b(?:not|never|\w+n['’]t)\s+(?:(?:yet|already|been|be|ever)\s+)*signed\b"
        r"|\bno\s+(?:agreement|contract|document)\s+"
        r"(?:(?:has|have|had|is|was|been|ever|yet)\s+)*signed\b"
        r"|\b(?:without|not(?:\s+to)?|never)\s+claim(?:ing)?\b"
        r"(?:(?!\b(?:but|and)\b)[^.;!?])*?\bsigned\b"
        r"|\bsigned\s+no\s+(?:agreements?|contracts?|documents?)\b"
    )
    remaining = re.sub(negated, "", fact, flags=re.IGNORECASE)
    return re.search(r"\bsigned\b", remaining, re.IGNORECASE) is not None
