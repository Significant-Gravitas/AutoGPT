"""Content checks: block seeded and patterned secrets without leaking them."""

from __future__ import annotations

import pytest

from .content_checks import (
    check_metadata,
    check_skill_bundle,
    check_skill_content,
    safe_diagnostic,
)

SEEDED = "hunter2-very-secret-value-9931"


def test_clean_skill_passes() -> None:
    body = "## Steps\n1. Run the import with {{API_KEY}}\n2. Check the row count\n"
    assert check_skill_content(body) is None


@pytest.mark.parametrize(
    "line,pattern_class",
    [
        ("token: ghp_" + "a" * 40, "github_token"),
        ("export AWS_KEY=AKIA" + "B" * 16, "aws_access_key"),
        ("-----BEGIN RSA PRIVATE KEY-----", "private_key"),
        ("Authorization: Bearer " + "x" * 30, "bearer_token"),
        ("postgres://app:supersecretpw@db.internal/app", "connection_string_password"),
        ("api_key = 'QmFzZTY0U2VjcmV0VmFsdWU='", "assigned_secret"),
    ],
)
def test_known_credential_patterns_are_blocked(line: str, pattern_class: str) -> None:
    failure = check_skill_content(f"## Steps\n1. First\n2. {line}\n")
    assert failure is not None
    assert failure.pattern_class == pattern_class
    assert failure.step == "section 1 › step 2 › line 3"


def test_diagnostic_never_contains_the_secret_even_from_a_heading() -> None:
    body = f"## Use key {SEEDED}\n1. do it\n"
    failure = check_skill_content(body, seeded_values=[SEEDED])
    assert failure is not None
    assert failure.pattern_class == "seeded_secret"
    assert SEEDED not in failure.describe()
    assert SEEDED not in failure.step
    assert failure.step == "section 1 › line 1"


def test_seeded_secret_in_a_bundle_filename_is_not_echoed() -> None:
    files = {"SKILL.md": "## Steps\n1. ok\n", f"references/{SEEDED}.md": SEEDED}
    failure = check_skill_bundle(files, seeded_values=[SEEDED])
    assert failure is not None
    assert SEEDED not in failure.describe()
    assert failure.file == "bundle file 2 (references/)"


def test_only_typed_upper_case_placeholders_are_exempt() -> None:
    assert (
        check_skill_content("Set ${SLACK_TOKEN} and {{API_KEY}} and <API_KEY>") is None
    )
    wrapped = "use <sk-" + "z" * 24 + "> as the key"
    failure = check_skill_content(wrapped)
    assert failure is not None
    assert failure.pattern_class == "openai_style_key"


def test_allowance_is_scoped_and_never_covers_seeded_values() -> None:
    body = "Authorization: Bearer " + "x" * 30
    assert check_skill_content(body, allowed_pattern_classes=["bearer_token"]) is None
    failure = check_skill_content(
        SEEDED, seeded_values=[SEEDED], allowed_pattern_classes=["seeded_secret"]
    )
    assert failure is not None


def test_metadata_fields_are_checked_before_anything_is_persisted() -> None:
    failure = check_metadata({"name": "sk-" + "q" * 24, "description": "fine"})
    assert failure is not None
    assert failure.file == "metadata field 1 (name)"


def test_safe_diagnostic_withholds_secret_bearing_text() -> None:
    assert safe_diagnostic("model said no") == "model said no"
    withheld = safe_diagnostic("first 200 chars: token: ghp_" + "b" * 40)
    assert "ghp_" not in withheld
    assert "withheld" in withheld
