import pytest

from backend.data.credential_scope import SCOPE_RULES, ServerTarget, scope_error

SMTP = SCOPE_RULES["smtp"]
DATABASE = SCOPE_RULES["database"]


@pytest.mark.parametrize(
    "scope,target,expected",
    [
        ("smtp://smtp.example.com", ("smtp", "smtp.example.com", 587), True),
        ("smtp://smtp.example.com", ("smtp", "SMTP.Example.com", 587), True),
        ("smtp://smtp.example.com", ("smtp", "smtp.example.com", None), True),
        ("SMTP://SMTP.Example.com:587", ("smtp", "smtp.example.com", 587), True),
        ("smtp://203.0.113.7", ("smtp", "203.0.113.7", 2525), True),
        ("smtp://[2001:db8::1]", ("smtp", "2001:db8::1", 587), True),
        ("smtp://[2001:db8::1]:587", ("smtp", "[2001:db8::1]", 587), True),
        # Another server, or another port when the scope names one
        ("smtp://smtp.example.com", ("smtp", "other.example.com", 587), False),
        ("smtp://smtp.example.com", ("smtp", "sub.smtp.example.com", 587), False),
        ("smtp://smtp.example.com:465", ("smtp", "smtp.example.com", 587), False),
        ("smtp://[2001:db8::1]:587", ("smtp", "2001:db8::1", 2525), False),
        # The block's host is compared as the literal it connects to
        (
            "smtp://smtp.example.com",
            ("smtp", "203.0.113.7 @smtp.example.com", 587),
            False,
        ),
        ("smtp://smtp.example.com", ("smtp", "smtp://smtp.example.com", 587), False),
        ("smtp://smtp.example.com", ("smtp", "smtp.example.com.", 587), False),
        ("smtp://smtp.example.com", ("smtp", "", 587), False),
        # Another scheme never matches
        ("smtp://smtp.example.com", ("postgresql", "smtp.example.com", 587), False),
    ],
)
def test_smtp_scope_matches(scope: str, target: tuple, expected: bool):
    scheme, host, port = target
    assert (
        SMTP.matches(scope, ServerTarget(scheme=scheme, host=host, port=port))
        is expected
    )


@pytest.mark.parametrize(
    "scope,target,expected",
    [
        ("postgresql://db.example.com", ("postgresql", "db.example.com", 5432), True),
        ("mysql://db.example.com:3306", ("mysql", "db.example.com", 3306), True),
        ("mssql://db.example.com", ("mssql", "db.example.com", 1433), True),
        # The engine is part of the scope
        ("postgresql://db.example.com", ("mysql", "db.example.com", 3306), False),
        ("mysql://db.example.com", ("postgresql", "db.example.com", 5432), False),
        (
            "postgresql://db.example.com:5432",
            ("postgresql", "db.example.com", 15432),
            False,
        ),
        # An SMTP scope is never a database scope
        ("smtp://db.example.com", ("smtp", "db.example.com", 587), False),
    ],
)
def test_database_scope_matches(scope: str, target: tuple, expected: bool):
    scheme, host, port = target
    assert (
        DATABASE.matches(scope, ServerTarget(scheme=scheme, host=host, port=port))
        is expected
    )


@pytest.mark.parametrize(
    "scope",
    [
        "smtp.example.com",
        "smtp.example.com:587",
        "smtp://",
        "smtp://smtp.example.com/",
        "smtp://smtp.example.com/path",
        "smtp://user@smtp.example.com",
        "smtp://user:pass@smtp.example.com",
        "smtp://*.example.com",
        "smtp://smtp.example.com:0",
        "smtp://smtp.example.com:70000",
        "smtp://smtp.example.com:abc",
        "smtp://smtp.example.com?x=1",
        "smtp://smtp.example.com#x",
        "smtp://2001:db8::1",
        "smtp://smtp example.com",
        "smtps://smtp.example.com",
        "postgresql://db.example.com",
    ],
)
def test_malformed_or_foreign_smtp_scope_rejected_and_matches_nothing(scope: str):
    assert scope_error("smtp", scope) is not None
    assert not SMTP.matches(
        scope, ServerTarget(scheme="smtp", host="smtp.example.com", port=587)
    )


@pytest.mark.parametrize(
    "provider,scope,ok",
    [
        ("smtp", "smtp://smtp.example.com:587", True),
        ("database", "postgresql://db.example.com:5432", True),
        ("database", "mysql://db.example.com", True),
        ("database", "mssql://[2001:db8::1]:1433", True),
        # New logins for these providers must name their server
        ("smtp", None, False),
        ("smtp", "", False),
        ("database", "   ", False),
        ("database", "smtp://db.example.com", False),
        ("database", "postgres://db.example.com", False),
        # Providers with a fixed server take no scope, but saving one is
        # refused rather than silently kept
        ("webshare_proxy", None, True),
        ("webshare_proxy", "", False),
        ("webshare_proxy", "smtp://smtp.example.com", False),
    ],
)
def test_scope_error(provider: str, scope: str | None, ok: bool):
    assert (scope_error(provider, scope) is None) is ok


@pytest.mark.parametrize(
    "provider,scope,message",
    [
        (
            "smtp",
            "smtp://smtp.example.com:99999",
            "Enter a valid server, like smtp.example.com or smtp.example.com:587 "
            "(port 1-65535)",
        ),
        (
            "smtp",
            "smtp://smtp.example.com/path",
            "Enter a valid server, like smtp.example.com or smtp.example.com:587 "
            "(port 1-65535)",
        ),
        (
            "database",
            "postgresql://db.example.com:0",
            "Enter a valid server, like db.example.com or db.example.com:5432 "
            "(port 1-65535)",
        ),
        (
            "smtp",
            None,
            "Enter the server this login is for, like smtp.example.com or "
            "smtp.example.com:587",
        ),
        (
            "database",
            "",
            "Enter the server this login is for, like db.example.com or "
            "db.example.com:5432",
        ),
    ],
)
def test_scope_error_speaks_of_the_server(
    provider: str, scope: str | None, message: str
):
    assert scope_error(provider, scope) == message
