"""A name that uses a joiner the way its script does is a name like any other.

The folder dialog left out every folder whose name held a character of a
Unicode `C` category. That takes in U+200C (zero width non-joiner), written
inside many Persian words, and U+200D (zero width joiner), which is how Indic
scripts ask for a conjunct and how an emoji sequence (a family, a rainbow
flag) is built. Those folders didn't show, so a person couldn't choose them.

The filter is there because the names go into a dialog a person reads: a
bidi control reorders what is shown, and a zero-width character shows as
nothing, so `Documents` followed by one looks like a second `Documents`.
That stays: a joiner where it only hides is still refused.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from autogpt_local_executor import hidden_characters
from autogpt_local_executor.computer_use import catalog
from autogpt_local_executor.computer_use.catalog import ArgumentRefused
from autogpt_local_executor.directory_browser import DirectoryBrowser, _valid_wire_name

ROOT = Path(__file__).resolve().parent.parent
ZWNJ, ZWJ, VS16 = "‌", "‍", "️"

# U+200C inside a Persian word: "I want" and "books".
MIKHAHAM = "می‌خواهم"
KETABHA = "کتاب‌ها"
# Emoji sequences, U+200D between pictographs.
FAMILY = "\U0001f468‍\U0001f469‍\U0001f467"
RAINBOW_FLAG = "\U0001f3f3️‍\U0001f308"

AS_ITS_SCRIPT_WRITES_IT = {
    "persian-non-joiner": MIKHAHAM,
    "persian-plural": KETABHA,
    "persian-among-other-words": f"پوشه‌های من 2024 {MIKHAHAM}",
    "devanagari-joiner-after-virama": "क्‍ष",
    "devanagari-non-joiner-after-virama": "क्‌ष",
    "malayalam-final-consonant": "അവന്‍",
    "sinhala-conjunct": "ශ්‍රී",
    "family": FAMILY,
    "family-in-a-name": f"Photos {FAMILY} 2024",
    "rainbow-flag": RAINBOW_FLAG,
    "rainbow-flag-without-its-selector": "\U0001f3f3‍\U0001f308",
    "pirate-flag": "\U0001f3f4‍☠️",
    "skin-tone-then-joiner": "\U0001f469\U0001f3fd‍\U0001f4bb",
    "heart-on-fire": "❤️‍\U0001f525",
    "health-worker": "\U0001f468‍⚕️",
    "black-cat": "\U0001f408‍⬛",
    "head-shaking": "\U0001f642‍↔️",
    "kiss-with-skin-tones": ("\U0001f468\U0001f3fb‍❤️‍\U0001f48b‍\U0001f468\U0001f3fc"),
}

# The bidi controls: each reorders, or claims the direction of, what is shown.
BIDI_CONTROLS = [
    *(chr(code) for code in range(0x202A, 0x202F)),
    *(chr(code) for code in range(0x2066, 0x206A)),
    "‎",
    "‏",
    "؜",
]

ONLY_HIDES = {
    "latin-non-joiner-at-the-end": f"Documents{ZWNJ}",
    "latin-joiner-at-the-end": f"Documents{ZWJ}",
    "latin-non-joiner-in-the-middle": f"Docu{ZWNJ}ments",
    "latin-joiner-in-the-middle": f"Docu{ZWJ}ments",
    "only-a-non-joiner": ZWNJ,
    "only-a-joiner": ZWJ,
    "non-joiner-at-the-start": f"{ZWNJ}می",
    "joiner-at-the-start": f"{ZWJ}\U0001f468",
    "persian-non-joiner-at-the-end": f"کتاب{ZWNJ}",
    "two-non-joiners": f"کتاب{ZWNJ}{ZWNJ}ها",
    "joiner-beside-a-non-joiner": f"کتاب{ZWJ}{ZWNJ}ها",
    # Reh doesn't join the letter after it, so a non-joiner there changes nothing.
    "after-a-letter-that-joins-nothing": f"کار{ZWNJ}ها",
    "arabic-then-latin": f"کتاب{ZWNJ}abc",
    "cyrillic": f"Док{ZWNJ}ум",
    "han": f"文{ZWJ}档",
    "pictograph-then-letter": f"\U0001f468{ZWJ}a",
    "letter-then-pictograph": f"a{ZWJ}\U0001f468",
    "joiner-after-the-last-pictograph": f"{FAMILY}{ZWJ}",
    "two-joiners-between-pictographs": f"\U0001f468{ZWJ}{ZWJ}\U0001f469",
    "non-joiner-between-pictographs": f"\U0001f468{ZWNJ}\U0001f469",
    "joiner-between-plus-signs": f"c+{ZWJ}+",
    "joiner-between-letterlike-symbols": f"©{ZWJ}®",
    "joiner-after-a-selector-on-a-letter": f"a{VS16}{ZWJ}\U0001f468",
    "zero-width-space": "Documents​",
    "word-joiner": "Docu⁠ments",
    "byte-order-mark": "﻿Documents",
    "soft-hyphen": "Docu­ments",
    "control": "bell\x07",
    "delete": "del\x7f",
    "private-use": "logo",
    "unassigned": "odd͸",
    "tag-characters": "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",
    **{f"bidi-U+{ord(control):04X}": f"evil{control}gpj.exe" for control in BIDI_CONTROLS},
    # A joiner used as its script does excuses nothing else in the name.
    **{
        f"bidi-U+{ord(control):04X}-in-persian": f"{MIKHAHAM}{control}" for control in BIDI_CONTROLS
    },
}

PLAIN = ["Documents", "my docs", "Документы", "文档"]
PLAIN += ["❤️ photos", "\U0001f1fa\U0001f1f8 trip", "café", "کتاب"]


@pytest.mark.parametrize("name", AS_ITS_SCRIPT_WRITES_IT.values(), ids=AS_ITS_SCRIPT_WRITES_IT)
def test_a_joiner_used_as_its_script_uses_it_hides_nothing(name: str) -> None:
    assert not hidden_characters.has_hidden_character(name)
    assert _valid_wire_name(name)


@pytest.mark.parametrize("name", ONLY_HIDES.values(), ids=ONLY_HIDES)
def test_a_character_that_only_hides_is_still_refused(name: str) -> None:
    assert hidden_characters.has_hidden_character(name)
    assert not _valid_wire_name(name)


@pytest.mark.parametrize("name", PLAIN)
def test_a_name_without_them_is_as_it_was(name: str) -> None:
    assert not hidden_characters.has_hidden_character(name)
    assert _valid_wire_name(name)


def test_a_lone_surrogate_is_still_no_wire_name() -> None:
    assert hidden_characters.has_hidden_character("bad\udc80")
    assert not _valid_wire_name("bad\udc80")
    assert not _valid_wire_name("")
    assert not _valid_wire_name("nul\x00")


def test_every_bidi_control_is_one_of_the_refused_categories() -> None:
    """The list above is the one the docs give; none may slip into an
    exception made for another character."""
    assert len(BIDI_CONTROLS) == 12
    for control in BIDI_CONTROLS:
        assert hidden_characters.has_hidden_character(control)
        assert hidden_characters.has_hidden_character(f"a{control}b")


# ── The folder dialog, on real folders ───────────────────────────────────────


def _made(parent: Path, names: dict[str, str]) -> dict[str, str]:
    """The folders of `names` this filesystem took, by the same keys. Some
    filesystems refuse a name others take (HFS+ has no name that is only a
    character it ignores)."""
    made = {}
    for key, name in names.items():
        try:
            (parent / name).mkdir()
        except (OSError, ValueError):
            continue
        made[key] = name
    return made


def _picker(home: Path) -> DirectoryBrowser:
    """A folder dialog whose one root is `home`. On Linux the roots also
    take in the removable disks of whoever runs the tests (`/media/<login
    name>`), each of which is looked at before the roots are given: a disk
    of theirs that had stopped answering held these tests, which are about
    names. None are looked for here."""
    browser = DirectoryBrowser(platform_name="linux", home=home)
    browser._linux_user_mounts = lambda: []  # type: ignore[method-assign]
    return browser


def test_the_dialogs_roots_here_are_the_home_folder_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    looked: list[Path] = []
    children_of = DirectoryBrowser._children_of

    def seen(parent: Path) -> list[Path]:
        looked.append(parent)
        return children_of(parent)

    monkeypatch.setattr(DirectoryBrowser, "_children_of", staticmethod(seen))

    roots = _picker(tmp_path).list_directories(None, None)

    assert len(roots.entries) == 1
    assert [parent for parent in looked if parent.parts[1:2] in (("media",), ("run",))] == []


def test_the_folder_dialog_lists_them_and_they_can_be_chosen(tmp_path: Path) -> None:
    shown = {key: AS_ITS_SCRIPT_WRITES_IT[key] for key in ("persian-non-joiner", "family")}
    shown |= {"rainbow-flag": RAINBOW_FLAG, "plain": "Documents"}
    left_out = {
        key: ONLY_HIDES[key]
        for key in (
            "latin-non-joiner-at-the-end",
            "latin-non-joiner-in-the-middle",
            "only-a-non-joiner",
            "bidi-U+202E",
        )
    }
    made_shown = _made(tmp_path, shown)
    made_left_out = _made(tmp_path, left_out)
    assert {"persian-non-joiner", "family", "plain"} <= set(made_shown)
    assert made_left_out, "this filesystem took none of the names that hide something"
    browser = _picker(tmp_path)

    roots = browser.list_directories(None, None)
    listing = browser.list_directories(roots.browse_id, roots.entries[0].directory_ref)

    listed = {entry.name: entry for entry in listing.entries}
    assert set(listed) == set(made_shown.values())
    for name in made_left_out.values():
        assert name not in listed
    # Each listed folder can be opened, and chosen for a chat.
    for name, entry in listed.items():
        inside = browser.list_directories(roots.browse_id, entry.directory_ref)
        assert inside.current is not None and inside.current.name == name
        chosen = browser.resolve(roots.browse_id, entry.directory_ref)
        assert chosen.name == name
        assert chosen.is_dir()


def test_a_folder_inside_one_with_a_joiner_is_reached(tmp_path: Path) -> None:
    (tmp_path / MIKHAHAM / "inner").mkdir(parents=True)
    browser = _picker(tmp_path)

    roots = browser.list_directories(None, None)
    top = browser.list_directories(roots.browse_id, roots.entries[0].directory_ref)
    [persian] = top.entries
    inside = browser.list_directories(roots.browse_id, persian.directory_ref)

    assert [entry.name for entry in inside.entries] == ["inner"]
    assert browser.resolve(roots.browse_id, inside.entries[0].directory_ref).name == "inner"


# ── The other places that refuse a name by its characters ────────────────────


@pytest.mark.parametrize(
    "name",
    [
        # "Word processor" in Persian, with its non-joiner.
        "واژه‌پرداز",
        f"Family {FAMILY}",
    ],
    ids=["persian", "emoji-sequence"],
)
def test_an_app_can_be_named_the_way_a_folder_can(name: str) -> None:
    """launch_app refused a name with any `C`-category character in it, so an
    app with a Persian name couldn't be launched by its name."""
    catalog.check_arguments("launch_app", {"name": name}, platform="darwin")


@pytest.mark.parametrize(
    "name",
    [f"Safari{ZWNJ}", f"Saf{ZWJ}ari", "evil‮irafaS", "tab\tname", "Safari​"],
    ids=["non-joiner-at-the-end", "joiner-in-the-middle", "bidi-override", "tab", "zero-width"],
)
def test_an_app_name_that_hides_something_is_still_refused(name: str) -> None:
    with pytest.raises(ArgumentRefused) as refused:
        catalog.check_arguments("launch_app", {"name": name}, platform="darwin")

    assert refused.value.argument == "name"


@pytest.mark.parametrize(
    "address",
    [
        # A host with a non-joiner: a browser may drop it before it looks the
        # host up, so the address it loads isn't the one that was checked.
        f"https://{MIKHAHAM}.example/",
        f"https://example.com/{FAMILY}",
        f"https://example.com/a{ZWNJ}b",
    ],
)
def test_a_web_address_takes_no_joiner_at_all(address: str) -> None:
    assert not catalog.is_web_address(address)


# ── What the package needs for it, and what the docs say ─────────────────────


def test_the_package_declares_what_the_rule_is_read_from() -> None:
    """The joining types come with the `idna` package, which httpx brings;
    a module that imports it says so in the package's own dependencies."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert any(
        dependency.replace(" ", "").startswith("idna>=") for dependency in project["dependencies"]
    )


def _text(document: str) -> str:
    return " ".join((ROOT / "docs" / document).read_text(encoding="utf-8").split())


def test_the_docs_say_which_names_the_dialog_lists() -> None:
    for document in ("PROTOCOL.md", "CROSS_PLATFORM.md"):
        text = _text(document)
        assert "U+200C" in text and "U+200D" in text, document
        assert "RFC 5892" in text, document
    protocol = _text("PROTOCOL.md")
    assert "U+202A to U+202E" in protocol and "U+2066 to U+2069" in protocol
    assert "tag characters" in protocol
