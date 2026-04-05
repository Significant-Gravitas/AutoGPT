"""Set up the AI and its goals"""

import logging
from typing import Optional

from autogpt.app.config import AppConfig

from forge.config.ai_directives import AIDirectives
from forge.config.ai_profile import AIProfile
from forge.logging.utils import print_attribute

logger = logging.getLogger(__name__)


def apply_overrides_to_ai_settings(
    ai_profile: AIProfile,
    directives: AIDirectives,
    override_name: Optional[str] = "",
    override_role: Optional[str] = "",
    replace_directives: bool = False,
    resources: Optional[list[str]] = None,
    constraints: Optional[list[str]] = None,
    best_practices: Optional[list[str]] = None,
):
    if override_name:
        ai_profile.ai_name = override_name
    if override_role:
        ai_profile.ai_role = override_role

    if replace_directives:
        if resources:
            directives.resources = resources
        if constraints:
            directives.constraints = constraints
        if best_practices:
            directives.best_practices = best_practices
    else:
        if resources:
            directives.resources += resources
        if constraints:
            directives.constraints += constraints
        if best_practices:
            directives.best_practices += best_practices


async def interactively_revise_ai_settings(
    ai_profile: AIProfile,
    directives: AIDirectives,
    app_config: AppConfig,
):
    """Print AI settings and return them.

    Args:
        ai_profile (AIConfig): The current AI profile.
        ai_directives (AIDirectives): The current AI directives.
        app_config (Config): The application configuration.

    Returns:
        AIConfig: The AI settings.
    """
    logger = logging.getLogger("revise_ai_profile")

    print_ai_settings(
        title="AI Settings",
        ai_profile=ai_profile,
        directives=directives,
        logger=logger,
    )
    logger.info(
        "To customize, use CLI args: --ai-name, --ai-role, "
        "--constraint, --resource, --best-practice"
    )

    return ai_profile, directives


def print_ai_settings(
    ai_profile: AIProfile,
    directives: AIDirectives,
    logger: logging.Logger,
    title: str = "AI Settings",
):
    print_attribute(title, "")
    print_attribute("-" * len(title), "")
    print_attribute("Name :", ai_profile.ai_name)
    print_attribute("Role :", ai_profile.ai_role)

    print_attribute("Constraints:", "" if directives.constraints else "(none)")
    for constraint in directives.constraints:
        logger.info(f"- {constraint}")
    print_attribute("Resources:", "" if directives.resources else "(none)")
    for resource in directives.resources:
        logger.info(f"- {resource}")
    print_attribute("Best practices:", "" if directives.best_practices else "(none)")
    for best_practice in directives.best_practices:
        logger.info(f"- {best_practice}")
