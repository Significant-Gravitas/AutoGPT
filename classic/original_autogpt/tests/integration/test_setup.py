import pytest
from autogpt.app.config import AppConfig
from autogpt.app.setup import (
    apply_overrides_to_ai_settings,
    interactively_revise_ai_settings,
)

from forge.config.ai_directives import AIDirectives
from forge.config.ai_profile import AIProfile


@pytest.mark.asyncio
async def test_apply_overrides_to_ai_settings():
    ai_profile = AIProfile(ai_name="Test AI", ai_role="Test Role")
    directives = AIDirectives(
        resources=["Resource1"],
        constraints=["Constraint1"],
        best_practices=["BestPractice1"],
    )

    apply_overrides_to_ai_settings(
        ai_profile,
        directives,
        override_name="New AI",
        override_role="New Role",
        replace_directives=True,
        resources=["NewResource"],
        constraints=["NewConstraint"],
        best_practices=["NewBestPractice"],
    )

    assert ai_profile.ai_name == "New AI"
    assert ai_profile.ai_role == "New Role"
    assert directives.resources == ["NewResource"]
    assert directives.constraints == ["NewConstraint"]
    assert directives.best_practices == ["NewBestPractice"]


@pytest.mark.asyncio
async def test_interactively_revise_ai_settings(config: AppConfig):
    """Test that interactively_revise_ai_settings returns the settings unchanged.

    The function was simplified to just print settings and return them without
    interactive prompts. Users should use CLI args to customize instead.
    """
    ai_profile = AIProfile(ai_name="Test AI", ai_role="Test Role")
    directives = AIDirectives(
        resources=["Resource1"],
        constraints=["Constraint1"],
        best_practices=["BestPractice1"],
    )

    returned_profile, returned_directives = await interactively_revise_ai_settings(
        ai_profile, directives, config
    )

    # Function returns the original settings unchanged
    assert returned_profile.ai_name == "Test AI"
    assert returned_profile.ai_role == "Test Role"
    assert returned_directives.resources == ["Resource1"]
    assert returned_directives.constraints == ["Constraint1"]
    assert returned_directives.best_practices == ["BestPractice1"]
