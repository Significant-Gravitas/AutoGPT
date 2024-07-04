
from __future__ import annotations
from typing import Dict
from ..registry import action
from forge.sdk import ForgeLogger, PromptEngine
from forge.llm import chat_completion_request
import os
from forge.sdk import Agent, LocalWorkspace
import re
import subprocess
import json
import test_code

LOG = ForgeLogger(__name__)
CodeType = Dict[str, str]
TestCaseType = Dict[str, str]


ARGO_TOML_CONTENT = """
[package]
name = "my_anchor_program"
version = "0.1.0"
edition = "2018"

[dependencies]
anchor-lang = "0.30.1"
"""

ANCHOR_TOML_CONTENT = """
[programs.localnet]
my_anchor_program = "4d3d5ab7f6b5e4b2b7d1f5d6e4b7d1f5d6e4b7d1"
"""
ERROR_INFO = ""


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
    global ERROR_INFO
    test_code.cargo_test_agbenchmark_config()  

   
    prompt_engine = PromptEngine("gpt-4o")
    lib_prompt = prompt_engine.load_prompt("anchor-lib", specification=specification, error_info=ERROR_INFO)
    instructions_prompt = prompt_engine.load_prompt("anchor-instructions", specification=specification, error_info=ERROR_INFO)
    errors_prompt = prompt_engine.load_prompt("anchor-errors", specification=specification, error_info=ERROR_INFO)
    cargo_toml_prompt = prompt_engine.load_prompt("anchor-cargo-toml", specification=specification, error_info=ERROR_INFO)
    anchor_toml_prompt = prompt_engine.load_prompt("anchor-anchor-toml", specification=specification, error_info=ERROR_INFO)
    
    messages = [
        {"role": "system", "content": "You are a code generation assistant specialized in Anchor for Solana."},
        {"role": "user", "content": lib_prompt},
        {"role": "user", "content": instructions_prompt},
        {"role": "user", "content": errors_prompt},
        {"role": "user", "content": cargo_toml_prompt},
        {"role": "user", "content": anchor_toml_prompt},
        {"role": "user", "content": "Return the whole code as a string with the file markers intact that you received in each of the input without changing their wording at all and use // becore comments."}, 

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

    base_path = agent.workspace.base_path if isinstance(
        agent.workspace, LocalWorkspace) else str(agent.workspace.base_path)
    project_path = os.path.join(base_path, task_id)
    LOG.info(f"Base path: {base_path}")
    LOG.info(f"Project path: {project_path}")

    LOG.info(f"id: {task_id}")
    LOG.info(f"Parts: {response_content}")

    file_actions = [
        ('Cargo.toml', ARGO_TOML_CONTENT),
        ('Anchor.toml', parts['Anchor.toml']),
        ('src/errors.rs', parts['errors.rs']),
        ('src/instructions.rs', parts['anchor-instructions.rs']),
        ('src/lib.rs', parts['anchor-lib.rs']),
    ]

    for file_path, file_content in file_actions:
        full_file_path = os.path.join(project_path, file_path)
        
        if os.path.exists(full_file_path):
            print(f"{file_path} already exists. Skipping regeneration.")
        else:
            print(f"Generating {file_path}. Press 'y' to continue...")
            if input().strip().lower() != 'y':
                return f"Generation halted by user at {file_path}."
            
            await agent.abilities.run_action(task_id, "write_file", file_path=full_file_path, data=file_content.encode())
            print(f"{file_path} generated successfully.")

            # Compile the generated file
            compile_result = await compile_file(agent, task_id, project_path, file_path)
            if "error" in compile_result.lower():
                LOG.error(f"Compilation failed for {file_path}: {compile_result}")
                print(f"Compilation failed for {file_path}, regenerating...")
                
                # Update ERROR_INFO with the compilation error
                ERROR_INFO = compile_result
                
                # Regenerate only the faulty file
                return await generate_solana_code(agent, task_id, specification)

    test_result = await agent.abilities.run_action(task_id, "test_code", project_path=project_path)
    if "All tests passed" not in test_result:
        LOG.info(f"Regenerating code due to errors: {test_result}")
        ERROR_INFO = test_result  # Update ERROR_INFO with the test error
        return await generate_solana_code(agent, task_id, specification)

    return "Solana on-chain code generated, tested, and verified successfully."


async def compile_file(agent: Agent, task_id: str, project_path: str, file_path: str) -> str:
    try:
        result = subprocess.run(['cargo', 'check', '--release'], cwd=project_path, capture_output=True, text=True)
        if result.returncode != 0:
            return result.stderr
        return "Compilation successful."
    except Exception as e:
        return f"Compilation failed: {e}"



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
        return "Failed to generate Solana on-chain code due to response parsing error."

    project_path = os.path.join(agent.workspace.base_path, task_id)

    await agent.abilities.run_action(
        task_id, "write_file", file_path=os.path.join(project_path, 'src', 'index.html'), data=parts['index.html'].encode()
    )
    await agent.abilities.run_action(
        task_id, "write_file", file_path=os.path.join(project_path, 'src', 'styles.css'), data=parts['styles.css'].encode()
    )
    await agent.abilities.run_action(
        task_id, "write_file", file_path=os.path.join(project_path, 'src', 'app.js'), data=parts['app.js'].encode()
    )
    await agent.abilities.run_action(
        task_id, "write_file", file_path=os.path.join(project_path, 'package.json'), data=parts['package.json'].encode()
    )
    await agent.abilities.run_action(
        task_id, "write_file", file_path=os.path.join(project_path, 'webpack.config.js'), data=parts['webpack.config.js'].encode()
    )

    return "Modular frontend code generated and written to respective files."


@action(
    name="generate_unit_tests",
    description="Generates unit tests for Solana code.",
    parameters=[
        {
            "name": "code_dict",
            "description": "Dictionary containing file names and respective code generated.",
            "type": "dict",
            "required": True
        }
    ],
    output_type="str",
)
async def generate_test_cases(agent: Agent, task_id: str, code_dict: CodeType) -> str:
    prompt_engine = PromptEngine("gpt-3.5-turbo")
    test_struct_prompt = prompt_engine.load_prompt("test-case-struct-return")

    messages = [
        {"role": "system", "content": "You are a code generation assistant specialized in generating test cases."},
    ]

    for file_name, code in code_dict.items():
        LOG.info(f"File Name: {file_name}")
        LOG.info(f"Code: {code}")
        test_prompt = prompt_engine.load_prompt(
            "test-case-generation", file_name=file_name, code=code)
        messages.append({"role": "user", "content": test_prompt})

    messages.append({"role": "user", "content": test_struct_prompt})

    chat_completion_kwargs = {
        "messages": messages,
        "model": "gpt-3.5-turbo",
    }

    chat_response = await chat_completion_request(**chat_completion_kwargs)
    response_content = chat_response["choices"][0]["message"]["content"]

    LOG.info(f"Response content: {response_content}")

    base_path = agent.workspace.base_path if isinstance(
        agent.workspace, LocalWorkspace) else str(agent.workspace.base_path)
    project_path = os.path.join(base_path, task_id)

    try:
        test_cases = parse_test_cases_response(response_content)
    except Exception as e:
        LOG.error(f"Error parsing test cases response: {e}")
        return "Failed to generate test cases due to response parsing error."

    for file_name, test_case in test_cases.items():
        test_file_path = os.path.join(project_path, 'tests', file_name)
        await agent.abilities.run_action(
            task_id, "write_file", file_path=test_file_path, data=test_case.encode()
        )

    return "Test cases generated and written to respective files."


def sanitize_json_string(json_string: str) -> str:
    # Replace newlines and tabs with escaped versions
    sanitized_string = json_string.replace(
        '\n', '\\n').replace('\t', '\\t').replace('    ', '\\t')
    return sanitized_string


def parse_test_cases_response(response_content: str) -> TestCaseType:
    try:
        # Extract JSON part from response content
        json_start = response_content.index('{')
        json_end = response_content.rindex('}') + 1
        json_content = response_content[json_start:json_end]

        # Sanitize JSON content
        sanitized_content = sanitize_json_string(json_content)

        # Load JSON content
        response_dict = json.loads(sanitized_content)

        file_name = response_dict["file_name"]
        test_file = response_dict["test_file"]

        # Unescape newlines and tabs in test_file
        test_file = test_file.replace('\\n', '\n').replace(
            '\\t', '\t').strip().strip('"')

        test_cases = {file_name: test_file}
        return test_cases
    except (json.JSONDecodeError, ValueError) as e:
        LOG.error(f"Error decoding JSON response: {e}")
        raise


def parse_response_content(response_content: str) -> dict:
    # This function will split the response content into different parts
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



