"""Tokenisation shared by the sources (tags) and the index (BM25 + names).

Deliberately tiny: lower-case, split CamelCase and punctuation, drop a few
stop words and the word "block" (every block carries it), and trim common
English suffixes so "issues", "generation" and "generator" meet "issue" and
"generate".  The stemmer is not linguistically correct; it only has to map
both sides of a match to the same string.  No embeddings.
"""

import re

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "the",
        "to",
        "for",
        "of",
        "in",
        "on",
        "with",
        "by",
        "from",
        "or",
        "is",
        "it",
        "this",
        "that",
        "be",
        "as",
        "at",
        "block",
        "blocks",
        "tool",
        "use",
        "using",
    }
)

# Query-side expansion for the platform's own vocabulary.  Applied to query
# tokens only, so documents stay as written.  Plain words here; both sides
# are stemmed below.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "if": ("condition",),
    "else": ("condition",),
    "python": ("code", "execute"),
    "javascript": ("code",),
    "bash": ("code", "execute", "shell"),
    "shell": ("bash", "execute"),
    "run": ("execute",),
    "execute": ("run",),
    "email": ("mail", "gmail"),
    "mail": ("email",),
    "llm": ("ai",),
    "gpt": ("ai", "openai"),
    "spreadsheet": ("sheet",),
    "sheet": ("spreadsheet",),
    "stringify": ("json", "encode"),
    "count": ("length",),
    "fetch": ("get", "read"),
    # Saving, storing and writing are one intent spread over three entries
    # (FileStoreBlock, write_workspace_file, memory_store); without the link
    # "save a file" reached neither of the first two.
    "save": ("store", "write"),
    "store": ("save", "write"),
    "write": ("save", "store"),
    "scrape": ("extract", "crawl"),
    "webpage": ("website", "web", "page"),
    "http": ("web",),
    "api": ("request",),
    "tweet": ("twitter",),
}

_SUFFIXES = ("ing", "ed", "al", "ion", "or", "er")
_MIN_STEM = 4


def split_camel(text: str) -> str:
    return _CAMEL.sub(" ", text)


def stem(token: str) -> str:
    token = _strip_plural(token)
    # Two rounds so "conditional" -> "condition" -> "condit" meets "condition".
    for _ in range(2):
        token = _strip_suffix(token)
    if len(token) > _MIN_STEM and token.endswith("e"):
        token = token[:-1]
    return token


def _strip_suffix(token: str) -> str:
    for suffix in _SUFFIXES:
        if len(token) - len(suffix) >= _MIN_STEM and token.endswith(suffix):
            return _collapse_double(token[: -len(suffix)])
    return token


def _strip_plural(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("sses", "ches", "shes", "xes", "zes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _collapse_double(token: str) -> str:
    """ "formatt" -> "format" after a suffix was stripped ("ll"/"ss" stay)."""
    if len(token) > 3 and token[-1] == token[-2] and token[-1] not in "lsaeiou":
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    words = _NON_ALNUM.split(split_camel(text).lower())
    return [stem(w) for w in words if w and w not in STOP_WORDS]


_SYNONYM_STEMS: dict[str, tuple[str, ...]] = {
    stem(word): tuple(stem(s) for s in synonyms) for word, synonyms in SYNONYMS.items()
}


def query_groups(text: str) -> list[list[str]]:
    """Distinct query tokens, each with its synonyms: one group per concept
    so coverage counts a concept once however it was matched."""
    return [
        [token, *_SYNONYM_STEMS.get(token, ())]
        for token in dict.fromkeys(tokenize(text))
    ]


def normalize_name(text: str) -> str:
    """Canonical form for exact-name matching: letters and digits only."""
    return re.sub(r"[^a-z0-9]", "", text.lower())
