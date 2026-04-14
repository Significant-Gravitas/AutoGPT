"""
Auto Documentation Block — generates README, API docs, architecture diagrams,
and inline code comments from source code.

Produces:
- README.md with installation, usage, and API reference
- Mermaid architecture diagrams
- OpenAPI-style endpoint documentation
- Inline docstring generation prompts
"""

import json
import logging
import os
import subprocess
from enum import Enum
from pathlib import Path
from typing import Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField

logger = logging.getLogger(__name__)


class DocType(str, Enum):
    README = "readme"
    API_REFERENCE = "api_reference"
    ARCHITECTURE_DIAGRAM = "architecture_diagram"
    INLINE_DOCSTRINGS = "inline_docstrings"
    CHANGELOG = "changelog"
    CONTRIBUTING = "contributing"


class AutoDocsInput(BlockSchemaInput):
    doc_type: DocType = SchemaField(
        default=DocType.README,
        description="Type of documentation to generate.",
    )
    project_name: str = SchemaField(
        default="",
        description="Project name.",
    )
    project_description: str = SchemaField(
        default="",
        description="Brief project description.",
    )
    source_code: str = SchemaField(
        default="",
        description="Source code to document (for inline docstrings or API reference).",
    )
    source_path: str = SchemaField(
        default="",
        description="Path to project directory (for README or architecture diagram).",
    )
    tech_stack: list = SchemaField(
        default_factory=list,
        description="Technologies used (e.g., ['Python', 'FastAPI', 'PostgreSQL']).",
    )
    existing_readme: str = SchemaField(
        default="",
        description="Existing README content to update (optional).",
    )
    api_endpoints: list = SchemaField(
        default_factory=list,
        description="List of API endpoint dicts for API reference generation.",
    )
    output_file: str = SchemaField(
        default="",
        description="Optional path to write the generated documentation.",
    )
    language: str = SchemaField(
        default="python",
        description="Programming language of the source code.",
    )


class AutoDocsOutput(BlockSchemaOutput):
    documentation: str = SchemaField(description="Generated documentation content.")
    doc_prompt: str = SchemaField(description="LLM prompt for generating the documentation.")
    output_file: str = SchemaField(description="Path where documentation was written (if any).")
    status: str = SchemaField(description="Operation status.")


def _scan_project_structure(source_path: str) -> dict:
    """Scan a project directory and return structure info."""
    path = Path(source_path)
    if not path.exists():
        return {}

    info = {
        "has_requirements": (path / "requirements.txt").exists(),
        "has_pyproject": (path / "pyproject.toml").exists(),
        "has_package_json": (path / "package.json").exists(),
        "has_docker": (path / "Dockerfile").exists() or (path / "docker-compose.yml").exists(),
        "has_github_actions": (path / ".github" / "workflows").exists(),
        "has_tests": any(path.rglob("test_*.py")) or any(path.rglob("*.test.ts")),
        "python_files": len(list(path.rglob("*.py"))),
        "ts_files": len(list(path.rglob("*.ts"))) + len(list(path.rglob("*.tsx"))),
        "directories": [d.name for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")],
    }

    # Try to read package.json for project info
    pkg_json = path / "package.json"
    if pkg_json.exists():
        try:
            pkg = json.loads(pkg_json.read_text())
            info["package_name"] = pkg.get("name", "")
            info["package_version"] = pkg.get("version", "")
            info["package_description"] = pkg.get("description", "")
        except Exception:
            pass

    return info


def _build_readme_prompt(
    project_name: str,
    description: str,
    tech_stack: list,
    project_info: dict,
    existing_readme: str,
) -> str:
    stack_str = ", ".join(tech_stack) if tech_stack else "Not specified"
    info_str = json.dumps(project_info, indent=2) if project_info else "N/A"

    base = f"""Generate a comprehensive, professional README.md for the following project.

## Project Details
- **Name**: {project_name}
- **Description**: {description}
- **Tech Stack**: {stack_str}

## Project Structure Info
```json
{info_str}
```
"""
    if existing_readme:
        base += f"\n## Existing README (update/improve this)\n```markdown\n{existing_readme[:3000]}\n```\n"

    base += """
## README Requirements
Generate a complete README.md with these sections:
1. **Project Title & Badges** (build status, license, version)
2. **Overview** — what the project does and why
3. **Features** — bullet list of key features
4. **Architecture** — brief architecture overview with Mermaid diagram
5. **Prerequisites** — system requirements
6. **Installation** — step-by-step setup instructions
7. **Configuration** — environment variables and config options
8. **Usage** — code examples and CLI commands
9. **API Reference** — if applicable
10. **Development** — how to run tests and contribute
11. **Deployment** — Docker/production deployment instructions
12. **License**

Use proper Markdown formatting with code blocks, tables, and badges.
Return ONLY the README.md content.
"""
    return base


def _build_architecture_diagram_prompt(
    project_name: str,
    description: str,
    tech_stack: list,
    source_code: str,
) -> str:
    stack_str = ", ".join(tech_stack) if tech_stack else "Not specified"
    code_snippet = source_code[:3000] if source_code else ""
    return f"""Generate a Mermaid architecture diagram for the following project.

## Project: {project_name}
## Description: {description}
## Tech Stack: {stack_str}

## Code Sample
```
{code_snippet}
```

## Requirements
1. Create a Mermaid flowchart (graph TD) showing the system architecture
2. Include: user/client, frontend, backend, database, external services, and integrations
3. Use clear labels and arrows showing data flow
4. Add a second diagram showing the agent workflow (if this is an AI agent project)

Return ONLY the Mermaid diagram code (starting with ```mermaid).
"""


def _build_docstring_prompt(source_code: str, language: str) -> str:
    return f"""Add comprehensive docstrings to all functions, classes, and methods
in the following {language} code that are missing them.

## Source Code
```{language}
{source_code}
```

## Requirements
- Use Google-style docstrings for Python (Args, Returns, Raises, Example sections)
- Use JSDoc for JavaScript/TypeScript
- Keep existing docstrings, only add missing ones
- Include type information, parameter descriptions, and return value descriptions
- Add module-level docstring if missing

Return ONLY the complete updated source code with docstrings added.
"""


class AutoDocsBlock(Block):
    """
    Generates documentation for code projects.

    Produces README files, API references, architecture diagrams (Mermaid),
    inline docstrings, changelogs, and contributing guides.
    Outputs a doc_prompt that should be passed to the LLM block.
    """

    class Input(AutoDocsInput):
        pass

    class Output(AutoDocsOutput):
        pass

    def __init__(self):
        super().__init__(
            id="f2a3b4c5-d6e7-8901-fab2-234567890123",
            description=(
                "Generates README, API docs, architecture diagrams, and docstrings. "
                "Outputs LLM prompts for documentation generation."
            ),
            categories={BlockCategory.DEVELOPER_TOOLS, BlockCategory.AI},
            input_schema=AutoDocsBlock.Input,
            output_schema=AutoDocsBlock.Output,
            test_input={
                "doc_type": DocType.README.value,
                "project_name": "TestProject",
                "project_description": "A test project.",
                "tech_stack": ["Python", "FastAPI"],
            },
            test_output=[
                ("status", "README prompt generated. Pass doc_prompt to LLM block."),
            ],
        )

    def run(self, input_data: Input, *, execution_stats=None, **kwargs) -> BlockOutput:
        project_info = {}
        if input_data.source_path:
            project_info = _scan_project_structure(input_data.source_path)

        if input_data.doc_type == DocType.README:
            prompt = _build_readme_prompt(
                input_data.project_name,
                input_data.project_description,
                input_data.tech_stack,
                project_info,
                input_data.existing_readme,
            )
            yield "documentation", ""
            yield "doc_prompt", prompt
            yield "output_file", input_data.output_file
            yield "status", "README prompt generated. Pass doc_prompt to LLM block."

        elif input_data.doc_type == DocType.ARCHITECTURE_DIAGRAM:
            prompt = _build_architecture_diagram_prompt(
                input_data.project_name,
                input_data.project_description,
                input_data.tech_stack,
                input_data.source_code,
            )
            yield "documentation", ""
            yield "doc_prompt", prompt
            yield "output_file", input_data.output_file
            yield "status", "Architecture diagram prompt generated."

        elif input_data.doc_type == DocType.INLINE_DOCSTRINGS:
            if not input_data.source_code:
                yield "documentation", ""
                yield "doc_prompt", ""
                yield "output_file", ""
                yield "status", "source_code is required for INLINE_DOCSTRINGS."
                return
            prompt = _build_docstring_prompt(input_data.source_code, input_data.language)
            yield "documentation", ""
            yield "doc_prompt", prompt
            yield "output_file", input_data.output_file
            yield "status", "Docstring generation prompt ready."

        elif input_data.doc_type == DocType.API_REFERENCE:
            endpoints_str = json.dumps(input_data.api_endpoints, indent=2) if input_data.api_endpoints else ""
            code_snippet = input_data.source_code[:4000] if input_data.source_code else ""
            prompt = f"""Generate a comprehensive API reference document for {input_data.project_name}.

## API Endpoints
```json
{endpoints_str}
```

## Source Code (excerpt)
```{input_data.language}
{code_snippet}
```

Generate a Markdown API reference with:
- Endpoint URL, method, description
- Request parameters (path, query, body) with types and descriptions
- Response schema with examples
- Authentication requirements
- Error codes and messages
- cURL and Python code examples for each endpoint

Return ONLY the Markdown API reference.
"""
            yield "documentation", ""
            yield "doc_prompt", prompt
            yield "output_file", input_data.output_file
            yield "status", "API reference prompt generated."

        elif input_data.doc_type == DocType.CHANGELOG:
            prompt = f"""Generate a CHANGELOG.md for {input_data.project_name}.

Project description: {input_data.project_description}

## Source Code Context
```
{input_data.source_code[:2000] if input_data.source_code else 'N/A'}
```

Generate a CHANGELOG.md following Keep a Changelog format (https://keepachangelog.com):
- Use semantic versioning
- Include sections: Added, Changed, Deprecated, Removed, Fixed, Security
- Start with [Unreleased] section
- Include example entries based on the project's features

Return ONLY the CHANGELOG.md content.
"""
            yield "documentation", ""
            yield "doc_prompt", prompt
            yield "output_file", input_data.output_file
            yield "status", "Changelog prompt generated."

        elif input_data.doc_type == DocType.CONTRIBUTING:
            prompt = f"""Generate a CONTRIBUTING.md guide for {input_data.project_name}.

Tech stack: {', '.join(input_data.tech_stack) if input_data.tech_stack else 'Not specified'}

Include:
1. Code of Conduct reference
2. How to report bugs
3. How to suggest features
4. Development setup instructions
5. Coding standards and style guide
6. Testing requirements
7. Pull request process
8. Commit message conventions (Conventional Commits)
9. Branch naming conventions

Return ONLY the CONTRIBUTING.md content.
"""
            yield "documentation", ""
            yield "doc_prompt", prompt
            yield "output_file", input_data.output_file
            yield "status", "Contributing guide prompt generated."

        else:
            yield "documentation", ""
            yield "doc_prompt", ""
            yield "output_file", ""
            yield "status", f"Unknown doc_type: {input_data.doc_type}"
