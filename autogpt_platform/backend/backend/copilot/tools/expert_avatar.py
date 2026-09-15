"""Generated expert avatars: a shape, a colour and one accessory.

The frontend draws these from a site-relative ``/avatars/<shape>.<color>.<accessory>.svg``
URL, so an expert raised by Otto gets a face without an upload. The id
lists and the name hash mirror ``components/molecules/BotAvatar/helpers.ts``
so a name seeds the same look on both sides.
"""

AVATAR_SHAPES = ["round", "dome", "squircle", "wide", "bean"]
AVATAR_COLORS = [
    "lavender",
    "plum",
    "amber",
    "sky",
    "mint",
    "coral",
    "indigo",
    "butter",
]
AVATAR_ACCESSORIES = [
    "none",
    "glasses",
    "headset",
    "star",
    "bow",
    "badge",
    "crown",
    "propeller",
    "ears",
    "flower",
    "bowtie",
    "headband",
]

# Accent colour token family -> avatar colour, tuned so the accent and the
# face agree without the model having to pick twice.
_TOKEN_COLORS = {
    "rose": "plum",
    "red": "coral",
    "orange": "coral",
    "amber": "amber",
    "yellow": "butter",
    "lime": "mint",
    "green": "mint",
    "emerald": "mint",
    "teal": "sky",
    "cyan": "sky",
    "sky": "sky",
    "blue": "sky",
    "indigo": "indigo",
    "violet": "lavender",
    "fuchsia": "plum",
}


def avatar_color_for_token(token: str | None) -> str | None:
    if not token:
        return None
    return _TOKEN_COLORS.get(token.split("-", 1)[0])


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


def seeded_avatar(name: str) -> tuple[str, str, str]:
    """Shape / colour / accessory the frontend would seed for *name*."""
    random = _seeded_random(_hash_seed(name.lower()))
    shape = AVATAR_SHAPES[int(random() * len(AVATAR_SHAPES))]
    color = AVATAR_COLORS[int(random() * len(AVATAR_COLORS))]
    accessory = AVATAR_ACCESSORIES[int(random() * len(AVATAR_ACCESSORIES))]
    return shape, color, accessory


def build_avatar_url(
    name: str,
    *,
    shape: str | None = None,
    accessory: str | None = None,
    color_token: str | None = None,
) -> str:
    """Resolve the generated-avatar URL, seeding whatever was not chosen."""
    seeded_shape, seeded_color, seeded_accessory = seeded_avatar(name)
    return (
        f"/avatars/{shape or seeded_shape}."
        f"{avatar_color_for_token(color_token) or seeded_color}."
        f"{accessory or seeded_accessory}.svg"
    )
