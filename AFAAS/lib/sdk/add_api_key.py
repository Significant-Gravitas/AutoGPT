import os
import re

from dotenv import load_dotenv

from AFAAS.core.tools.builtins.user_interaction import user_interaction


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

    # Optionally, update the environment variables in your script
    os.environ[key] = value
