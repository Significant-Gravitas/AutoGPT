from __future__ import annotations

import os
import shutil
import subprocess
from unittest.mock import MagicMock, Mock, patch

import pytest
from dotenv import load_dotenv

from AFAAS.core.tools.tool import Tool
from AFAAS.lib.sdk.add_api_key import ensure_api_key, install_and_import_package
from AFAAS.plugins.tools.langchain_google_places import (
    GooglePlacesAPIWrapper,
    GooglePlacesTool,
)


@pytest.fixture(scope="module")
def setup_google_places_test():
    # Setup: Delete API key if exists and uninstall package
    backup_file = remove_api_key_if_exists("GPLACES_API_KEY")
    uninstall_package("googlemaps")

    yield

    # Teardown: Clean up after test
    remove_api_key_if_exists("GPLACES_API_KEY")
    uninstall_package("googlemaps")
    if backup_file is not None:
        # If the backup exists, restore it
        if os.path.exists(backup_file):
            shutil.copy2(backup_file, ".env")
            os.remove(backup_file)


def remove_api_key_if_exists(key: str, env_file_path=".env"):
    # Logic to remove the API key from the environment or .env file
    # Load the existing .env file
    load_dotenv()

    # Check if the key exists in the environment variables
    if key in os.environ:
        # Create a backup of the .env file
        backup_file = env_file_path + ".bak"
        shutil.copy2(env_file_path, backup_file)

        # Remove the key from the environment
        del os.environ[key]

        # Remove the key from the .env file
        with open(env_file_path, "r") as env_file:
            lines = env_file.readlines()

        with open(env_file_path, "w") as env_file:
            for line in lines:
                # Remove lines containing the key
                if not line.startswith(f"{key}="):
                    env_file.write(line)

        return backup_file

    return None


def uninstall_package(package_name: str):
    poetry_active = os.getenv("POETRY_ACTIVE") == "1"
    if poetry_active:
        subprocess.run(["poetry", "remove", package_name], check=False)
    else:
        subprocess.run(["pip", "uninstall", "-y", package_name], check=False)


def test_integration_google_places(setup_google_places_test):
    with patch("builtins.input", return_value="AIza-123456"):
        # patch('AFAAS.lib.sdk.add_api_key.input', return_value='AIza-123456')
        # patch('builtins.input', return_value='AIza-124456')
        ensure_api_key(
            key="GPLACES_API_KEY",
            api_name="Google Places (Google Maps API)",
            section="GOOGLE APIs",
        )
        load_dotenv()
        gplaces_api_key = os.getenv("GPLACES_API_KEY")

        assert gplaces_api_key == "AIza-123456"

        install_and_import_package("googlemaps")
        import googlemaps

        tool = Tool.generate_from_langchain_tool(
            langchain_tool=GooglePlacesTool(api_wrapper=GooglePlacesAPIWrapper()),
            categories=["search", "google", "maps"],
        )

        assert tool is not None
        assert tool.method is not None
