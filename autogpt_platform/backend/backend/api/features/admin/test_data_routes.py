import importlib.util
import logging
from enum import Enum
from pathlib import Path
from types import ModuleType

from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel

from backend.data.db import prisma
from backend.util.metrics import DiscordChannel, discord_send_alert
from backend.util.settings import AppEnvironment, BehaveAs, Settings

logger = logging.getLogger(__name__)
settings = Settings()

# The dev-only seeding scripts live in backend/test/, outside the importable
# package tree, so they are loaded by file path via importlib at request time.
_TEST_SCRIPT_DIR = Path(__file__).resolve().parents[4] / "test"


class TestDataScriptType(str, Enum):
    """Available test data generation scripts."""

    FULL = "full"  # test_data_creator.py - creates 100+ users, comprehensive data
    E2E = "e2e"  # e2e_test_data.py - creates a set of users with API functions


class GenerateTestDataRequest(BaseModel):
    """Request model for test data generation."""

    script_type: TestDataScriptType = TestDataScriptType.E2E


class GenerateTestDataResponse(BaseModel):
    """Response model for test data generation."""

    success: bool
    message: str
    details: dict | None = None


router = APIRouter(
    prefix="/admin",
    tags=["test-data"],
    dependencies=[Security(requires_admin_user)],
)


@router.post(
    "/generate-test-data",
    response_model=GenerateTestDataResponse,
    summary="Generate Test Data",
)
async def generate_test_data(
    request: GenerateTestDataRequest,
    admin_user_id: str = Security(get_user_id),
) -> GenerateTestDataResponse:
    """
    Generate test data for the platform.

    This endpoint runs the test data generation scripts to populate the database
    with sample users, agents, graphs, executions, store listings, and more.

    Available script types:
    - `e2e`: Creates a set of test users with graphs, library agents, presets,
             and store submissions. Uses API functions for better compatibility.
    - `full`: Creates 100+ users with comprehensive test data using direct Prisma
              calls. Generates more data but may take longer.

    **Warning**: This will add significant data to your database. Use with caution.
    **Note**: This endpoint is only available in local environments; requests from
    any shared/cloud environment are rejected and alerted on.
    """
    await _guard_local_only(admin_user_id, request.script_type)

    logger.info(
        f"Admin user {admin_user_id} is generating test data "
        f"with script type: {request.script_type}"
    )

    try:
        if request.script_type == TestDataScriptType.E2E:
            return await _run_e2e_generation()
        return await _run_full_generation()
    except Exception:
        logger.exception("Error generating test data")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate test data. Check the server logs for details.",
        )


async def _guard_local_only(
    admin_user_id: str, script_type: TestDataScriptType
) -> None:
    """Reject and alert on any test-data generation outside a local environment."""
    is_local = (
        settings.config.app_env == AppEnvironment.LOCAL
        and settings.config.behave_as == BehaveAs.LOCAL
    )
    if is_local:
        return

    logger.warning(
        "Test data generation blocked outside local environment. Admin: %s",
        admin_user_id,
    )
    alert_message = (
        f"🚨 **SECURITY ALERT**: Test data generation attempted outside a local "
        f"environment!\n"
        f"Admin User ID: `{admin_user_id}`\n"
        f"Environment: `{settings.config.app_env}` / `{settings.config.behave_as}`\n"
        f"Script Type: `{script_type}`\n"
        f"Action: Request was blocked."
    )
    try:
        await discord_send_alert(alert_message, DiscordChannel.PLATFORM)
    except Exception:
        logger.exception("Failed to send Discord alert for blocked test-data request")

    raise HTTPException(
        status_code=403,
        detail="Test data generation is only available in local environments.",
    )


async def _run_e2e_generation() -> GenerateTestDataResponse:
    """Run the E2E seeding script against the shared Prisma connection."""
    creator_class = getattr(_load_test_script("e2e_test_data"), "TestDataCreator")

    if not prisma.is_connected():
        await prisma.connect()

    creator = creator_class()
    await creator.create_all_test_data()

    return GenerateTestDataResponse(
        success=True,
        message="E2E test data generated successfully",
        details={
            "users_created": len(creator.users),
            "graphs_created": len(creator.agent_graphs),
            "library_agents_created": len(creator.library_agents),
            "store_submissions_created": len(creator.store_submissions),
            "presets_created": len(creator.presets),
            "api_keys_created": len(creator.api_keys),
        },
    )


async def _run_full_generation() -> GenerateTestDataResponse:
    """Run the comprehensive seeding script (owns its own Prisma lifecycle)."""
    run_full = getattr(_load_test_script("test_data_creator"), "main")
    await run_full()

    return GenerateTestDataResponse(
        success=True,
        message="Full test data generated successfully",
        details={
            "script": "test_data_creator.py",
            "note": "Created 100+ users with comprehensive test data",
        },
    )


def _load_test_script(module_name: str) -> ModuleType:
    """Load a dev-only seeding script from backend/test/ by file path."""
    script_path = _TEST_SCRIPT_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load test-data script '{module_name}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
