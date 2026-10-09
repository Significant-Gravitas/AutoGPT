import re

_SOURCE_TOKENS = re.compile(
    r""""(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\#[^\n]*|//[^\n]*|[\[\](){}"']""",
    re.DOTALL,
)
_DELIMITERS = {"(": ")", "[": "]", "{": "}"}


def validate_complete_delimiters(source: str) -> None:
    """Reject incomplete structure; the frontend still validates the full DSL."""
    stack: list[str] = []
    for match in _SOURCE_TOKENS.finditer(source):
        token = match.group()
        if token in {'"', "'"}:
            raise ValueError("Unclosed string. Complete it before rendering the view")
        if len(token) > 1 or token.startswith("#"):
            continue
        if token in _DELIMITERS:
            stack.append(token)
        elif not stack or _DELIMITERS[stack.pop()] != token:
            raise ValueError(
                f"Mismatched closing delimiter {token!r} at character {match.start()}. "
                "Complete each array, object and component call before rendering"
            )
    if stack:
        raise ValueError(
            f"Unclosed delimiter {stack[-1]!r}. Complete each array, object and "
            "component call before rendering the view"
        )
