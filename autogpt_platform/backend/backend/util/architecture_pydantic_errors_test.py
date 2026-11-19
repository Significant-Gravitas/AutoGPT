import ast

import pytest

from backend.util.architecture_test import BACKEND_ROOT, _iter_backend_py_files

# ---------------------------------------------------------------------------
# Rule: code raises no built-in pydantic error of its own
# ---------------------------------------------------------------------------
#
# Motivation: a log line keeps the schema's side of a built-in pydantic error's
# context, such as what was expected (`backend/api/utils/validation_errors.py`).
# A validator that raised `PydanticKnownError` with a context of its own could
# put the refused value there. Raise `ValueError` or `PydanticCustomError`
# instead, whose messages a log line leaves out.

_RENDERER = "api/utils/validation_errors.py"


def test_no_code_raises_a_built_in_pydantic_error():
    offenders = [
        f"  backend/{rel}:{lineno}"
        for py in _iter_backend_py_files()
        if not py.name.endswith("_test.py")
        and (rel := py.relative_to(BACKEND_ROOT).as_posix()) != _RENDERER
        for lineno in _known_error_uses(ast.parse(py.read_text(encoding="utf-8")))
    ]

    assert not offenders, (
        "A built-in pydantic error's context reaches the logs as written, so only "
        f"backend/{_RENDERER} may use `PydanticKnownError`. Raise `ValueError` or "
        "`PydanticCustomError` instead, and don't import or name it. These lines "
        "do:\n" + "\n".join(sorted(offenders))
    )


def _known_error_uses(tree: ast.Module) -> list[int]:
    """Lines that import or name `PydanticKnownError`."""
    return [
        node.lineno
        for node in ast.walk(tree)
        if (isinstance(node, ast.Name) and node.id == "PydanticKnownError")
        or (isinstance(node, ast.Attribute) and node.attr == "PydanticKnownError")
        or (
            isinstance(node, ast.ImportFrom)
            and any(alias.name == "PydanticKnownError" for alias in node.names)
        )
    ]


@pytest.mark.parametrize(
    "source, flagged",
    [
        ("from pydantic_core import PydanticKnownError", True),
        ('raise PydanticKnownError("literal_error", {"expected": value})', True),
        ('raise pydantic_core.PydanticKnownError("missing")', True),
        ('raise PydanticCustomError("refused", "Refused")', False),
        ('raise ValueError("refused")', False),
    ],
)
def test_the_scan_finds_each_use(source: str, flagged: bool):
    assert bool(_known_error_uses(ast.parse(source))) is flagged
