import logging
from enum import Enum

from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, Security
from pydantic import BaseModel

from backend.util.metrics import DiscordChannel, discord_send_alert
from backend.util.settings import AppEnvironment, Settings

logger = logging.getLogger(__name__)
settings = Settings()


class TestDataScriptType(str, Enum):
    """Available test data generation scripts."""

    FULL = "full"  # test_data_creator.py - creates 100+ users, comprehensive data
    E2E = "e2e"  # e2e_test_data.py - creates 15 users with API functions


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
    tags=["admin", "test-data"],
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
    - `e2e`: Creates 15 test users with graphs, library agents, presets, and store submissions.
             Uses API functions for better compatibility. (Recommended)
    - `full`: Creates 100+ users with comprehensive test data using direct Prisma calls.
              Generates more data but may take longer.

    **Warning**: This will add significant data to your database. Use with caution.
    **Note**: This endpoint is disabled in production environments.
    """
    # Block execution in production environment
    if settings.config.app_env == AppEnvironment.PRODUCTION:
        alert_message = (
            f"🚨 **SECURITY ALERT**: Test data generation attempted in PRODUCTION!\n"
            f"Admin User ID: `{admin_user_id}`\n"
            f"Script Type: `{request.script_type}`\n"
            f"Action: Request was blocked."
        )
        logger.warning(
            f"Test data generation blocked in production. Admin: {admin_user_id}"
        )

        # Send Discord alert
        try:
            await discord_send_alert(alert_message, DiscordChannel.PLATFORM)
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")

        return GenerateTestDataResponse(
            success=False,
            message="Test data generation is disabled in production environments.",
        )

    logger.info(
        f"Admin user {admin_user_id} is generating test data with script type: {request.script_type}"
    )

    try:
        if request.script_type == TestDataScriptType.E2E:
            # Import and run the E2E test data creator
            # We need to import within the function to avoid circular imports
            import sys
            from pathlib import Path

            from backend.data.db import prisma

            # Add the test directory to the path
            test_dir = Path(__file__).parent.parent.parent.parent.parent / "test"
            sys.path.insert(0, str(test_dir))

            try:
                from e2e_test_data import (  # pyright: ignore[reportMissingImports]
                    TestDataCreator,
                )

                # Connect to database if not already connected
                if not prisma.is_connected():
                    await prisma.connect()

                creator = TestDataCreator()
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
            finally:
                # Remove the test directory from the path
                if str(test_dir) in sys.path:
                    sys.path.remove(str(test_dir))

        elif request.script_type == TestDataScriptType.FULL:
            # Import and run the full test data creator
            import sys
            from pathlib import Path

            test_dir = Path(__file__).parent.parent.parent.parent.parent / "test"
            sys.path.insert(0, str(test_dir))

            try:
                import test_data_creator  # pyright: ignore[reportMissingImports]

                create_full_test_data = test_data_creator.main

                await create_full_test_data()

                return GenerateTestDataResponse(
                    success=True,
                    message="Full test data generated successfully",
                    details={
                        "script": "test_data_creator.py",
                        "note": "Created 100+ users with comprehensive test data",
                    },
                )
            finally:
                if str(test_dir) in sys.path:
                    sys.path.remove(str(test_dir))

        else:
            return GenerateTestDataResponse(
                success=False,
                message=f"Unknown script type: {request.script_type}",
            )

    except Exception as e:
        logger.exception(f"Error generating test data: {e}")
        return GenerateTestDataResponse(
            success=False,
            message=f"Failed to generate test data: {str(e)}",
        )
