"""Generated expert avatars: a Notion-style face, seeded from the name.

The frontend draws these from a site-relative
``/avatars/notion/<indices>.<color>.svg`` URL, so an expert raised by Otto gets
a face without an upload. The category list, part counts, "none" odds and the
name hash all mirror ``components/molecules/NotionAvatar/helpers.ts``, so a
name seeds the same look on both sides — there is a test asserting it.
"""

# Draw order, back to front. Must match NOTION_CATEGORIES on the frontend.
NOTION_CATEGORIES = [
    "face",
    "nose",
    "mouth",
    "eyes",
    "eyebrows",
    "glasses",
    "hair",
    "accessories",
    "details",
    "beard",
]

NOTION_PART_COUNTS = {
    "face": 16,
    "nose": 14,
    "mouth": 20,
    "eyes": 14,
    "eyebrows": 16,
    "glasses": 15,
    "hair": 59,
    "accessories": 15,
    "details": 14,
    "beard": 17,
}

# Index 0 draws nothing for these, and sampling uniformly would put glasses and
# a beard on almost every face.
_NONE_ODDS = {
    "glasses": 0.55,
    "beard": 0.6,
    "accessories": 0.7,
    "details": 0.65,
    "hair": 0.04,
}

# Named looks Otto can ask for, each a pool of parts that read that way. The
# specific part inside a pool stays seeded, so two experts wearing glasses do
# not end up in identical frames.
AVATAR_GLASSES = {
    "none": [0],
    "glasses": [1, 2, 3, 4, 5, 6, 14],
    "sunglasses": [7, 8, 9, 10, 11, 12],
}

AVATAR_BEARD = {
    "none": [0],
    "stubble": [2, 5, 12],
    "full": [1, 3, 4, 13],
    "moustache": [14, 15],
}

AVATAR_HAT = {
    "none": [0],
    "cap": [11, 12],
}

AVATAR_COLORS = [
    "rose",
    "red",
    "orange",
    "amber",
    "yellow",
    "lime",
    "green",
    "emerald",
    "teal",
    "cyan",
    "sky",
    "blue",
    "indigo",
    "violet",
    "fuchsia",
]


def avatar_color_for_token(token: str | None) -> str | None:
    """The accent family is the avatar colour, so no two accents share a face."""
    if not token:
        return None
    family = token.split("-", 1)[0]
    return family if family in AVATAR_COLORS else None


def _hash_seed(value: str) -> int:
    seed = 2166136261
    for char in value:
        seed ^= ord(char)
        seed = (seed * 16777619) & 0xFFFFFFFF
    return seed


def _seeded_random(seed: int):
    state = seed or 1

    def next_value() -> float:
        nonlocal state
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        return state / 4294967296

    return next_value


def seeded_avatar(name: str) -> tuple[dict[str, int], str]:
    """Parts and colour the frontend would seed for *name*.

    Draws in the same order and the same number of times as ``randomNotionConfig``:
    one value per always-drawn category, and for a category that can be absent,
    one value for the coin flip plus a second only when it lands on a part.
    """
    random = _seeded_random(_hash_seed(name.lower()))
    parts: dict[str, int] = {}
    for category in NOTION_CATEGORIES:
        count = NOTION_PART_COUNTS[category]
        none_odds = _NONE_ODDS.get(category)
        if none_odds is None:
            parts[category] = int(random() * count)
        elif random() < none_odds:
            parts[category] = 0
        else:
            parts[category] = 1 + int(random() * (count - 1))
    parts["details"] = 0
    color = AVATAR_COLORS[int(random() * len(AVATAR_COLORS))]
    return parts, color


def _pool_choice(name: str, category: str, pool: list[int]) -> int:
    """Deterministic pick within a named look, off the same name."""
    return pool[_hash_seed(f"{name.lower()}:{category}") % len(pool)]


def build_avatar_url(
    name: str,
    *,
    glasses: str | None = None,
    beard: str | None = None,
    hat: str | None = None,
    color_token: str | None = None,
) -> str:
    """Resolve the generated-avatar URL, seeding whatever was not chosen."""
    parts, seeded_color = seeded_avatar(name)

    for category, choice, options in (
        ("glasses", glasses, AVATAR_GLASSES),
        ("beard", beard, AVATAR_BEARD),
        ("accessories", hat, AVATAR_HAT),
    ):
        if choice and choice in options:
            parts[category] = _pool_choice(name, category, options[choice])

    color = avatar_color_for_token(color_token) or seeded_color
    indices = "-".join(str(parts[category]) for category in NOTION_CATEGORIES)
    return f"/avatars/notion/{indices}.{color}.svg"
