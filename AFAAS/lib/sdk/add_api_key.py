import os
import re
import subprocess
import sys

from dotenv import load_dotenv

from AFAAS.core.tools.builtins.user_interaction import user_interaction
from AFAAS.lib.sdk.logger import LOG


def add_to_env_file(key: str, query: str, section=None, env_file_path=".env"):
    """
    Add a new key-value pair to a .env file with optional sections.

    Args:
        key (str): The key to add.
        value (str): The value associated with the key.
        section (str, optional): The section to add the key-value pair within.
        env_file_path (str, optional): The path to the .env file. Default is ".env".
    """
    # Load the existing .env file
    load_dotenv()
    value = input(query)
    set_to_env_file(key = key,
                    value =  value,
                    section=section,
                    env_file_path=env_file_path
                    )
    # Optionally, update the environment variables in your script
    os.environ[key] = value
    return value

def set_to_env_file(key: str, value: str, section=None, env_file_path=".env"):
    # Define the new entry
    new_entry = f"{key}={value}"

    # Check if the section exists in the .env file
    if section:
        section_pattern = re.compile(rf"#+\s*###{re.escape(section)}\s*#+")
        with open(env_file_path, "r") as env_file:
            file_contents = env_file.read()

        if not section_pattern.search(file_contents):
            # Create the section if it doesn't exist
            with open(env_file_path, "a") as env_file:
                env_file.write(
                    f"\n\n#############################################################################\n"
                )
                env_file.write(f"### {section}\n")
                env_file.write(
                    f"#############################################################################\n"
                )

    # Append the new entry to the .env file
    with open(env_file_path, "a") as env_file:
        env_file.write("\n" + new_entry)



def ensure_api_key(key : str , api_name : str, section = "API KEYS"):
    """
    Ensure that an API key for a given API is available in the environment variables.
    If not, prompt the user to add it and update the environment.

    :param key: The environment variable key for the API key
    :param api_name: A descriptive name of the API (e.g., "Google Places (Google Maps API)")
    :param section: The section in the environment file where the key should be added (default "API KEYS")
    :return: The API key
    """
    api_key = os.getenv(key, None)

    if not api_key:
        LOG.warning(f"A {key} is required to use {api_name}")

        # Constructing the query message based on the API name
        query = f"To use {api_name} please insert your API Key:"

        # Assuming add_to_env_file is a function you've defined to add a key to an environment file
        api_key = add_to_env_file(key=key, query=query, section=section)
        os.environ[key] = api_key

    return api_key


def install_and_import_package(package_name):
    try:
        __import__(package_name)

    except ImportError:
        #exit("T1")
        LOG.info(f"{package_name} package is not installed. Installing...")
        #exit("T2")
        # Check if the script is running in a Poetry environment
        if os.getenv('POETRY_ACTIVE') == '1':
            #exit("T3")
            result = subprocess.run(["poetry", "add", package_name], check=False)
            #exit("T3B")
        else:
            #exit("T4")
            result = subprocess.run(["pip", "install", package_name], check=False)
            #exit("T4B")

        if result.returncode != 0:
            LOG.error(f"Failed to install the {package_name} package.")
            sys.exit(1)
        #sys.exit(result.returncode)
        LOG.info(f"{package_name} package has been installed.")
        __import__(package_name)
