"""Skills a roster expert ships with, installed when it is hired.

A roster template is not owned by anyone, and a skill is stored in its
owner's workspace, so a template cannot hold skill content the way a hired
expert can. `Expert.skills` on a template is therefore display text only:
without something to install the real files, hiring a marketplace expert
gives you its personality and none of its capability.

So the content lives here as checked-in markdown, one folder per roster
entry named after the template, and `hire_expert` copies it into the new
expert's own folder in the hiring user's workspace. Same shape as
`store/starter_skills/`, and parsed with the same
:func:`parse_skill_markdown` the upload endpoint uses, so a shipped skill
cannot drift from the format an installed one has.

Editing a skill here changes what *future* hires get. Existing hires own
their copies and are deliberately left alone: they may have been edited,
and silently overwriting someone's skill is worse than it being stale.
"""

import logging
from pathlib import Path

from backend.copilot.tools.skills import ParsedSkill, parse_skill_markdown

logger = logging.getLogger(__name__)

CONTENT_DIR = Path(__file__).parent / "roster_skills"


def skills_for_template(template_name: str) -> list[ParsedSkill]:
    """Parsed skills shipped with the roster entry called *template_name*.

    Empty when the entry ships none, which is the normal case for a roster
    expert whose value is its workflows rather than its skills.

    A file that does not parse is skipped with a warning rather than
    failing the hire: a malformed skill should cost that one skill, not the
    expert. The validation that stops it reaching here is
    ``roster_skills_test``.
    """
    folder = CONTENT_DIR / template_name.strip().lower()
    if not folder.is_dir():
        return []

    parsed: list[ParsedSkill] = []
    for path in sorted(folder.glob("*/SKILL.md")):
        skill = parse_skill_markdown(
            path.read_text(encoding="utf-8"), fallback_name=path.parent.name
        )
        if skill is None:
            logger.warning(
                "Roster skill %s/%s does not parse; skipping",
                template_name,
                path.parent.name,
            )
            continue
        parsed.append(skill)
    return parsed


def template_names_with_skills() -> list[str]:
    """Folder names under ``roster_skills/``. Used by the roster test to
    check every folder matches a real roster entry, so a renamed expert
    cannot silently stop shipping its skills."""
    if not CONTENT_DIR.is_dir():
        return []
    return sorted(p.name for p in CONTENT_DIR.iterdir() if p.is_dir())
