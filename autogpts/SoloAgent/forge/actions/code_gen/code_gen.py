from __future__ import annotations

from ..registry import action
from forge.sdk import ForgeLogger, PromptEngine
from forge.llm import chat_completion_request
import os
from forge.sdk import Agent, LocalWorkspace
import re
import subprocess

LOG = ForgeLogger(__name__)
MAX_ATTEMPTS = 3


@action(
    name="test_code",
    description="Test the generated code for errors",
    parameters=[
        {
            "name": "project_path",
            "description": "Path to the project directory",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def test_code(agent: Agent, task_id: str, project_path: str) -> str:
    try:
        result = subprocess.run(
            ['cargo', 'test'], cwd=project_path, capture_output=True, text=True)

        if result.returncode != 0:
            LOG.error(f"Test failed with errors: {result.stderr}")
            return result.stderr  # Return errors
        else:
            LOG.info(f"All tests passed: {result.stdout}")
            return "All tests passed"

    except Exception as e:
        LOG.error(f"Error testing code: {e}")
        return f"Failed to test code: {e}"


@action(
    name="generate_solana_code",
    description="Generate Solana on-chain code using Anchor based on the provided specification",
    parameters=[
        {
            "name": "specification",
            "description": "Code specification",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def generate_solana_code(agent: Agent, task_id: str, specification: str) -> str:
    prompt_engine = PromptEngine("gpt-4o")
    lib_prompt = prompt_engine.load_prompt(
        "anchor-lib", specification=specification)
    instructions_prompt = prompt_engine.load_prompt(
        "anchor-instructions", specification=specification)
    errors_prompt = prompt_engine.load_prompt(
        "anchor-errors", specification=specification)
    cargo_toml_prompt = prompt_engine.load_prompt(
        "anchor-cargo-toml", specification=specification)
    anchor_toml_prompt = prompt_engine.load_prompt(
        "anchor-anchor-toml", specification=specification)

    messages = [
        {"role": "system", "content": "You are a code generation assistant specialized in Anchor for Solana."},
        {"role": "user", "content": lib_prompt},
        {"role": "user", "content": instructions_prompt},
        {"role": "user", "content": errors_prompt},
        {"role": "user", "content": cargo_toml_prompt},
        {"role": "user", "content": anchor_toml_prompt},
        {"role": "user", "content": "Return the whole code as a string with the file markers intact that you received in each of the input without changing their wording at all, also remove the prompts in the files when generating."}
    ]

    chat_completion_kwargs = {
        "messages": messages,
        "model": "gpt-3.5-turbo",
    }

    chat_response = await chat_completion_request(**chat_completion_kwargs)
    response_content = chat_response["choices"][0]["message"]["content"]

    LOG.info(f"Response content: {response_content}")

    try:
        parts = parse_response_content(response_content)
    except Exception as e:
        LOG.error(f"Error parsing response content: {e}")
        return "Failed to generate Solana on-chain code due to response parsing error."

    project_base_path = agent.workspace.base_path if isinstance(
        agent.workspace, LocalWorkspace) else str(agent.workspace.base_path)
    project_path = os.path.join(
        project_base_path, 'solana_mvp_project', 'onchain', 'programs', 'my_anchor_program')
    src_path = os.path.join(project_path, 'src')

    # Create necessary directories
    os.makedirs(src_path, exist_ok=True)

    # Write each part to the respective file
    file_paths = {
        'anchor-lib.rs': os.path.join(src_path, 'lib.rs'),
        'anchor-instructions.rs': os.path.join(src_path, 'instructions.rs'),
        'errors.rs': os.path.join(src_path, 'errors.rs'),
        'Cargo.toml': os.path.join(project_path, 'Cargo.toml'),
        'Anchor.toml': os.path.join(project_path, 'Anchor.toml'),
    }

    for key, file_path in file_paths.items():
        if key in parts and parts[key]:
            with open(file_path, 'w') as file:
                file.write(parts[key])
        else:
            LOG.error(f"{key} content is missing.")

    # Run tests in the project directory
    test_result = await test_code(agent, task_id, project_path)
    attempts = 0
    while "All tests passed" not in test_result and attempts < MAX_ATTEMPTS:
        # Regenerate the code based on errors
        LOG.info(f"Regenerating code due to errors: {test_result}")
        test_result = await generate_solana_code(agent, task_id, specification)
        attempts += 1

    return "Solana on-chain code generated, tested, and verified successfully."


@action(
    name="generate_frontend_code",
    description="Generate frontend code based on the provided specification",
    parameters=[
        {
            "name": "specification",
            "description": "Frontend code specification",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def generate_frontend_code(agent, task_id: str, specification: str) -> str:
    prompt_engine = PromptEngine("gpt-3.5-turbo")
    index_prompt = prompt_engine.load_prompt(
        "frontend-index", specification=specification)
    styles_prompt = prompt_engine.load_prompt(
        "frontend-styles", specification=specification)
    app_prompt = prompt_engine.load_prompt(
        "frontend-app", specification=specification)
    package_json_prompt = prompt_engine.load_prompt(
        "frontend-package-json", specification=specification)
    webpack_config_prompt = prompt_engine.load_prompt(
        "frontend-webpack-config", specification=specification)

    messages = [
        {"role": "system", "content": "You are a code generation assistant specialized in frontend development."},
        {"role": "user", "content": index_prompt},
        {"role": "user", "content": styles_prompt},
        {"role": "user", "content": app_prompt},
        {"role": "user", "content": package_json_prompt},
        {"role": "user", "content": webpack_config_prompt},
    ]

    chat_completion_kwargs = {
        "messages": messages,
        "model": "gpt-3.5-turbo",
    }
    chat_response = await chat_completion_request(**chat_completion_kwargs)
    response_content = chat_response["choices"][0]["message"]["content"]

    try:
        parts = parse_response_content(response_content)
    except Exception as e:
        LOG.error(f"Error parsing response content: {e}")
        return "Failed to generate frontend code due to response parsing error."

    project_path = os.path.join(
        agent.workspace.path, 'solana_mvp_project', 'frontend')

    # Create necessary directories
    src_path = os.path.join(project_path, 'src')
    os.makedirs(src_path, exist_ok=True)

    file_paths = {
        'index.html': os.path.join(src_path, 'index.html'),
        'styles.css': os.path.join(src_path, 'styles.css'),
        'app.js': os.path.join(src_path, 'app.js'),
        'package.json': os.path.join(project_path, 'package.json'),
        'webpack.config.js': os.path.join(project_path, 'webpack.config.js'),
    }

    for key, file_path in file_paths.items():
        if key in parts and parts[key]:
            with open(file_path, 'w') as file:
                file.write(parts[key])
        else:
            LOG.error(f"{key} content is missing.")

    return "Modular frontend code generated and written to respective files."


def parse_response_content(response_content: str) -> dict:
    parts = {
        'anchor-lib.rs': '',
        'anchor-instructions.rs': '',
        'errors.rs': '',
        'Cargo.toml': '',
        'Anchor.toml': ''
    }

    current_part = None
    for line in response_content.split('\n'):
        if '// anchor-lib.rs' in line:
            current_part = 'anchor-lib.rs'
        elif '// anchor-instructions.rs' in line:
            current_part = 'anchor-instructions.rs'
        elif '// errors.rs' in line:
            current_part = 'errors.rs'
        elif '# Cargo.toml' in line:
            current_part = 'Cargo.toml'
        elif '# Anchor.toml' in line:
            current_part = 'Anchor.toml'
        elif current_part:
            parts[current_part] += line + '\n'

    for key in parts:
        parts[key] = re.sub(r'```|rust|toml', '', parts[key]).strip()

    return parts

