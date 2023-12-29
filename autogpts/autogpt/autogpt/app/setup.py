"""Set up the AI and its goals"""
import logging
from typing import Optional

from autogpt.app.utils import clean_input
from autogpt.config import AIDirectives, AIProfile, Config
from autogpt.logs.helpers import print_attribute

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
    app_config: Config,
):
    """Interactively revise the AI settings.

    Args:
        ai_profile (AIConfig): The current AI profile.
        ai_directives (AIDirectives): The current AI directives.
        app_config (Config): The application configuration.

    Returns:
        AIConfig: The revised AI settings.
    """
    logger = logging.getLogger("revise_ai_profile")

    revised = False

    while True:
        # Print the current AI configuration
        print_ai_settings(
            title="Current AI Settings" if not revised else "Revised AI Settings",
            ai_profile=ai_profile,
            directives=directives,
            logger=logger,
        )

        if (
            await clean_input(app_config, "Continue with these settings? [Y/n]")
            or app_config.authorise_key
        ) == app_config.authorise_key:
            break

        # Ask for revised ai_profile
        ai_profile.ai_name = (
            await clean_input(
                app_config, "Enter AI name (or press enter to keep current):"
            )
            or ai_profile.ai_name
        )
        ai_profile.ai_role = (
            await clean_input(
                app_config, "Enter new AI role (or press enter to keep current):"
            )
            or ai_profile.ai_role
        )

        # Revise constraints
        for i, constraint in enumerate(directives.constraints):
            print_attribute(f"Constraint {i+1}:", f'"{constraint}"')
            new_constraint = (
                await clean_input(
                    app_config,
                    f"Enter new constraint {i+1}"
                    " (press enter to keep current, or '-' to remove):",
                )
                or constraint
            )
            if new_constraint == "-":
                directives.constraints.remove(constraint)
            elif new_constraint:
                directives.constraints[i] = new_constraint

        # Add new constraints
        while True:
            new_constraint = await clean_input(
                app_config,
                "Press enter to finish, or enter a constraint to add:",
            )
            if not new_constraint:
                break
            directives.constraints.append(new_constraint)

        # Revise resources
        for i, resource in enumerate(directives.resources):
            print_attribute(f"Resource {i+1}:", f'"{resource}"')
            new_resource = (
                await clean_input(
                    app_config,
                    f"Enter new resource {i+1}"
                    " (press enter to keep current, or '-' to remove):",
                )
                or resource
            )
            if new_resource == "-":
                directives.resources.remove(resource)
            elif new_resource:
                directives.resources[i] = new_resource

        # Add new resources
        while True:
            new_resource = await clean_input(
                app_config,
                "Press enter to finish, or enter a resource to add:",
            )
            if not new_resource:
                break
            directives.resources.append(new_resource)

        # Revise best practices
        for i, best_practice in enumerate(directives.best_practices):
            print_attribute(f"Best Practice {i+1}:", f'"{best_practice}"')
            new_best_practice = (
                await clean_input(
                    app_config,
                    f"Enter new best practice {i+1}"
                    " (press enter to keep current, or '-' to remove):",
                )
                or best_practice
            )
            if new_best_practice == "-":
                directives.best_practices.remove(best_practice)
            elif new_best_practice:
                directives.best_practices[i] = new_best_practice

        # Add new best practices
        while True:
            new_best_practice = await clean_input(
                app_config,
                "Press enter to finish, or add a best practice to add:",
            )
            if not new_best_practice:
                break
            directives.best_practices.append(new_best_practice)

        revised = True

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
