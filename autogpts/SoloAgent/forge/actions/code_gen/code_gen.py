
from __future__ import annotations


from ..registry import action
from forge.sdk import ForgeLogger, PromptEngine
from forge.llm import chat_completion_request
import json
import os
LOG = ForgeLogger(__name__)


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
async def generate_solana_code(agent, task_id: str, specification: str) -> str:
    prompt_engine = PromptEngine("gpt-3.5-turbo")
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
    ]

    chat_completion_kwargs = {
        "messages": messages,
        "model": "gpt-3.5-turbo",
    }
    chat_response = await chat_completion_request(**chat_completion_kwargs)
    generated_code = json.loads(
        chat_response["choices"][0]["message"]["content"])

    project_path = os.path.join(
        agent.workspace.path, 'solana_mvp_project', 'onchain', 'programs', 'my_anchor_program')

    await agent.abilities.run_action(
        task_id, "write_file", file_path=os.path.join(project_path, 'src', 'lib.rs'), data=generated_code['lib.rs'].encode()
    )
    await agent.abilities.run_action(
        task_id, "write_file", file_path=os.path.join(project_path, 'src', 'instructions.rs'), data=generated_code['instructions.rs'].encode()
    )
    await agent.abilities.run_action(
        task_id, "write_file", file_path=os.path.join(project_path, 'src', 'errors.rs'), data=generated_code['errors.rs'].encode()
    )
    await agent.abilities.run_action(
        task_id, "write_file", file_path=os.path.join(project_path, 'Cargo.toml'), data=generated_code['Cargo.toml'].encode()
    )
    await agent.abilities.run_action(
        task_id, "write_file", file_path=os.path.join(project_path, 'Anchor.toml'), data=generated_code['Anchor.toml'].encode()
    )

    return "Modular Solana on-chain code generated with Anchor and written to respective files."


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
    generated_code = json.loads(
        chat_response["choices"][0]["message"]["content"])

    project_path = os.path.join(
        agent.workspace.path, 'solana_mvp_project', 'frontend')

    await agent.abilities.run_action(
        task_id, "write_file", file_path=os.path.join(project_path, 'src', 'index.html'), data=generated_code['index.html'].encode()
    )
    await agent.abilities.run_action(
        task_id, "write_file", file_path=os.path.join(project_path, 'src', 'styles.css'), data=generated_code['styles.css'].encode()
    )
    await agent.abilities.run_action(
        task_id, "write_file", file_path=os.path.join(project_path, 'src', 'app.js'), data=generated_code['app.js'].encode()
    )
    await agent.abilities.run_action(
        task_id, "write_file", file_path=os.path.join(project_path, 'package.json'), data=generated_code['package.json'].encode()
    )
    await agent.abilities.run_action(
        task_id, "write_file", file_path=os.path.join(project_path, 'webpack.config.js'), data=generated_code['webpack.config.js'].encode()
    )

    return "Modular frontend code generated and written to respective files."
