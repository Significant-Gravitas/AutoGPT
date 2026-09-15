"""Writing a package's skills into an expert's own folder.

Its own module because both ways an expert can arrive from a package need it
and neither may import the other: importing a file goes through
``package_import`` (which reads the expert DB), while hiring a published
template goes through ``experts_db`` itself.
"""

import logging

from backend.api.features.experts.package_model import ExpertPackage, PackagedSkill
from backend.copilot.tools.skills import parse_skill_markdown, store_user_skill

logger = logging.getLogger(__name__)


async def install_package_skills(
    user_id: str,
    expert_id: str,
    package: ExpertPackage,
    skills: list[PackagedSkill],
) -> list[str]:
    """Write each skill's whole folder into the expert's workspace, returning
    the names of the ones that did not make it.

    Best effort per skill, like every other install on this path: an expert
    missing one of its skills is far more use than no expert at all.
    ``store_user_skill`` records the name on the expert row itself, so a skill
    that fails leaves no name behind and the expert never lists one it does not
    have.
    """
    failed: list[str] = []
    for card in skills:
        stored = package.skills.get(card.slug)
        parsed = parse_skill_markdown(stored.skill_md) if stored else None
        if stored is None or parsed is None:
            failed.append(card.name)
            continue
        try:
            await store_user_skill(
                user_id,
                name=parsed.name or card.name,
                description=parsed.description,
                body=parsed.body,
                triggers=list(parsed.triggers),
                files=stored.files,
                expert_id=expert_id,
            )
        except Exception:
            logger.exception(
                f"Failed to install packaged skill {card.slug!r} on expert #{expert_id}"
            )
            failed.append(card.name)
    return failed
