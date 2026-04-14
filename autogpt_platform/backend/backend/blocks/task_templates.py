"""
Task Templates Block — saves and loads reusable prompt presets.

Templates are stored in a JSON file and can be parameterized with
variable substitution (e.g., {{project_name}}, {{language}}).
Built-in templates cover common coding tasks.
"""

import json
import logging
import re
from datetime import datetime, timezone
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

DEFAULT_TEMPLATES_FILE = "./data/task_templates.json"

BUILTIN_TEMPLATES = {
    "add_feature": {
        "name": "Add Feature",
        "description": "Add a new feature to an existing codebase.",
        "prompt": (
            "Add the following feature to the {{project_name}} project:\n\n"
            "Feature: {{feature_description}}\n\n"
            "Requirements:\n"
            "- Follow the existing code style and patterns\n"
            "- Write unit tests for the new feature\n"
            "- Update documentation if needed\n"
            "- Ensure backward compatibility\n\n"
            "Language/Stack: {{language}}"
        ),
        "variables": ["project_name", "feature_description", "language"],
        "persona": "fullstack_dev",
        "tags": ["feature", "development"],
    },
    "fix_bug": {
        "name": "Fix Bug",
        "description": "Diagnose and fix a reported bug.",
        "prompt": (
            "Fix the following bug in {{project_name}}:\n\n"
            "Bug Description: {{bug_description}}\n\n"
            "Steps to Reproduce:\n{{reproduction_steps}}\n\n"
            "Expected Behavior: {{expected_behavior}}\n\n"
            "Analyze the root cause, implement a fix, and add a regression test."
        ),
        "variables": ["project_name", "bug_description", "reproduction_steps", "expected_behavior"],
        "persona": "backend_dev",
        "tags": ["bugfix"],
    },
    "code_review": {
        "name": "Code Review",
        "description": "Review code for quality, security, and best practices.",
        "prompt": (
            "Review the following {{language}} code for:\n"
            "1. Correctness and logic errors\n"
            "2. Security vulnerabilities\n"
            "3. Performance issues\n"
            "4. Code style and readability\n"
            "5. Missing tests or documentation\n\n"
            "Code to review:\n```{{language}}\n{{code}}\n```\n\n"
            "Provide specific, actionable feedback with code examples."
        ),
        "variables": ["language", "code"],
        "persona": "code_reviewer",
        "tags": ["review", "quality"],
    },
    "generate_tests": {
        "name": "Generate Tests",
        "description": "Generate a comprehensive test suite for existing code.",
        "prompt": (
            "Generate a comprehensive {{framework}} test suite for the following {{language}} code.\n\n"
            "Module: {{module_name}}\n"
            "```{{language}}\n{{code}}\n```\n\n"
            "Requirements:\n"
            "- Cover all public functions/methods\n"
            "- Include edge cases and error conditions\n"
            "- Use descriptive test names\n"
            "- Add docstrings to each test\n"
            "- Use parametrize for data-driven tests"
        ),
        "variables": ["framework", "language", "module_name", "code"],
        "persona": "test_engineer",
        "tags": ["testing", "quality"],
    },
    "refactor": {
        "name": "Refactor Code",
        "description": "Refactor code for improved readability and maintainability.",
        "prompt": (
            "Refactor the following {{language}} code to improve:\n"
            "- Readability and clarity\n"
            "- Separation of concerns\n"
            "- Removal of duplication (DRY principle)\n"
            "- Performance where obvious\n\n"
            "Refactoring goals: {{goals}}\n\n"
            "Original code:\n```{{language}}\n{{code}}\n```\n\n"
            "Preserve all existing functionality. Include a brief explanation of changes."
        ),
        "variables": ["language", "goals", "code"],
        "persona": "code_reviewer",
        "tags": ["refactor", "quality"],
    },
    "generate_docs": {
        "name": "Generate Documentation",
        "description": "Generate README, API docs, or inline documentation.",
        "prompt": (
            "Generate {{doc_type}} documentation for the {{project_name}} project.\n\n"
            "Project description: {{project_description}}\n\n"
            "Include:\n"
            "- Overview and purpose\n"
            "- Installation instructions\n"
            "- Usage examples\n"
            "- API reference (if applicable)\n"
            "- Configuration options\n"
            "- Contributing guidelines"
        ),
        "variables": ["doc_type", "project_name", "project_description"],
        "persona": "documentation_writer",
        "tags": ["documentation"],
    },
    "security_audit": {
        "name": "Security Audit",
        "description": "Perform a security audit of code or infrastructure.",
        "prompt": (
            "Perform a security audit of the following {{target_type}} for {{project_name}}.\n\n"
            "Scope: {{audit_scope}}\n\n"
            "Check for:\n"
            "- Authentication and authorization flaws\n"
            "- Input validation and injection vulnerabilities\n"
            "- Sensitive data exposure\n"
            "- Insecure dependencies\n"
            "- Security misconfigurations\n"
            "- OWASP Top 10 issues\n\n"
            "Provide severity ratings (Critical/High/Medium/Low) and remediation steps."
        ),
        "variables": ["target_type", "project_name", "audit_scope"],
        "persona": "security_auditor",
        "tags": ["security", "audit"],
    },
    "setup_cicd": {
        "name": "Setup CI/CD",
        "description": "Create GitHub Actions CI/CD pipeline configuration.",
        "prompt": (
            "Create a GitHub Actions CI/CD pipeline for {{project_name}}.\n\n"
            "Stack: {{stack}}\n"
            "Requirements:\n{{requirements}}\n\n"
            "Include:\n"
            "- Lint and format checks\n"
            "- Unit and integration tests\n"
            "- Build and Docker image creation\n"
            "- Deployment to {{deployment_target}}\n"
            "- Branch protection rules recommendation"
        ),
        "variables": ["project_name", "stack", "requirements", "deployment_target"],
        "persona": "devops",
        "tags": ["devops", "cicd"],
    },
}


class TemplateOperation(str, Enum):
    SAVE = "save"
    LOAD = "load"
    LIST = "list"
    DELETE = "delete"
    RENDER = "render"
    LIST_BUILTIN = "list_builtin"


class TaskTemplatesInput(BlockSchemaInput):
    operation: TemplateOperation = SchemaField(
        default=TemplateOperation.LIST,
        description="Operation: save, load, list, delete, render, or list_builtin.",
    )
    template_id: str = SchemaField(
        default="",
        description="Template ID for load, delete, or render operations.",
    )
    template_name: str = SchemaField(
        default="",
        description="Human-readable template name (for save).",
    )
    template_prompt: str = SchemaField(
        default="",
        description="Template prompt text with {{variable}} placeholders (for save).",
    )
    template_description: str = SchemaField(
        default="",
        description="Template description (for save).",
    )
    variables: dict = SchemaField(
        default_factory=dict,
        description="Variable values for rendering: {'variable_name': 'value'}.",
    )
    persona: str = SchemaField(
        default="fullstack_dev",
        description="Default persona for this template.",
    )
    tags: list = SchemaField(
        default_factory=list,
        description="Tags for categorizing the template.",
    )
    templates_file: str = SchemaField(
        default=DEFAULT_TEMPLATES_FILE,
        description="Path to the JSON file storing templates.",
    )


class TaskTemplatesOutput(BlockSchemaOutput):
    rendered_prompt: str = SchemaField(description="Rendered prompt with variables substituted.")
    template_id: str = SchemaField(description="ID of the template.")
    templates: list = SchemaField(description="List of templates (for LIST operations).")
    status: str = SchemaField(description="Operation result status.")
    persona: str = SchemaField(description="Persona associated with the template.")


def _load_templates(templates_file: str) -> dict:
    path = Path(templates_file)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_templates(templates_file: str, templates: dict) -> None:
    path = Path(templates_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(templates, indent=2))


def _render_template(prompt: str, variables: dict) -> str:
    """Substitute {{variable}} placeholders in a template."""
    for key, value in variables.items():
        prompt = prompt.replace(f"{{{{{key}}}}}", str(value))
    return prompt


class TaskTemplatesBlock(Block):
    """
    Manages reusable task prompt templates with variable substitution.

    Save custom templates, load built-in presets (Add Feature, Fix Bug,
    Code Review, Security Audit, etc.), and render them with variable values.
    """

    class Input(TaskTemplatesInput):
        pass

    class Output(TaskTemplatesOutput):
        pass

    def __init__(self):
        super().__init__(
            id="d0e1f2a3-b4c5-6789-defa-012345678901",
            description=(
                "Manages reusable task prompt templates. Save custom templates, "
                "load built-in presets, and render with variable substitution."
            ),
            categories={BlockCategory.AI},
            input_schema=TaskTemplatesBlock.Input,
            output_schema=TaskTemplatesBlock.Output,
            test_input={
                "operation": TemplateOperation.LIST_BUILTIN.value,
                "templates_file": "/tmp/test_templates.json",
            },
            test_output=[
                ("status", f"Found {len(BUILTIN_TEMPLATES)} built-in templates."),
            ],
        )

    def run(self, input_data: Input, *, execution_stats=None, **kwargs) -> BlockOutput:
        user_templates = _load_templates(input_data.templates_file)

        if input_data.operation == TemplateOperation.LIST_BUILTIN:
            templates_list = [
                {
                    "id": tid,
                    "name": t["name"],
                    "description": t["description"],
                    "variables": t.get("variables", []),
                    "persona": t.get("persona", "fullstack_dev"),
                    "tags": t.get("tags", []),
                    "builtin": True,
                }
                for tid, t in BUILTIN_TEMPLATES.items()
            ]
            yield "rendered_prompt", ""
            yield "template_id", ""
            yield "templates", templates_list
            yield "status", f"Found {len(BUILTIN_TEMPLATES)} built-in templates."
            yield "persona", ""

        elif input_data.operation == TemplateOperation.LIST:
            all_templates = []
            for tid, t in BUILTIN_TEMPLATES.items():
                all_templates.append({
                    "id": tid, "name": t["name"], "description": t["description"],
                    "builtin": True, "persona": t.get("persona", ""), "tags": t.get("tags", []),
                })
            for tid, t in user_templates.items():
                all_templates.append({
                    "id": tid, "name": t.get("name", tid), "description": t.get("description", ""),
                    "builtin": False, "persona": t.get("persona", ""), "tags": t.get("tags", []),
                })
            yield "rendered_prompt", ""
            yield "template_id", ""
            yield "templates", all_templates
            yield "status", f"Found {len(all_templates)} templates ({len(BUILTIN_TEMPLATES)} built-in, {len(user_templates)} custom)."
            yield "persona", ""

        elif input_data.operation == TemplateOperation.SAVE:
            import uuid
            tid = input_data.template_id or str(uuid.uuid4())[:8]
            user_templates[tid] = {
                "name": input_data.template_name or tid,
                "description": input_data.template_description,
                "prompt": input_data.template_prompt,
                "persona": input_data.persona,
                "tags": input_data.tags,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            _save_templates(input_data.templates_file, user_templates)
            yield "rendered_prompt", ""
            yield "template_id", tid
            yield "templates", []
            yield "status", f"Template '{tid}' saved."
            yield "persona", input_data.persona

        elif input_data.operation == TemplateOperation.LOAD:
            tid = input_data.template_id
            template = BUILTIN_TEMPLATES.get(tid) or user_templates.get(tid)
            if not template:
                yield "rendered_prompt", ""
                yield "template_id", tid
                yield "templates", []
                yield "status", f"Template '{tid}' not found."
                yield "persona", ""
                return
            yield "rendered_prompt", template.get("prompt", "")
            yield "template_id", tid
            yield "templates", [template]
            yield "status", f"Template '{tid}' loaded."
            yield "persona", template.get("persona", "fullstack_dev")

        elif input_data.operation == TemplateOperation.RENDER:
            tid = input_data.template_id
            template = BUILTIN_TEMPLATES.get(tid) or user_templates.get(tid)
            if not template:
                yield "rendered_prompt", ""
                yield "template_id", tid
                yield "templates", []
                yield "status", f"Template '{tid}' not found."
                yield "persona", ""
                return
            rendered = _render_template(template.get("prompt", ""), input_data.variables)
            # Check for unresolved variables
            unresolved = re.findall(r"\{\{(\w+)\}\}", rendered)
            status = f"Template '{tid}' rendered."
            if unresolved:
                status += f" Warning: unresolved variables: {unresolved}"
            yield "rendered_prompt", rendered
            yield "template_id", tid
            yield "templates", []
            yield "status", status
            yield "persona", template.get("persona", "fullstack_dev")

        elif input_data.operation == TemplateOperation.DELETE:
            tid = input_data.template_id
            if tid in user_templates:
                del user_templates[tid]
                _save_templates(input_data.templates_file, user_templates)
                yield "rendered_prompt", ""
                yield "template_id", tid
                yield "templates", []
                yield "status", f"Template '{tid}' deleted."
                yield "persona", ""
            elif tid in BUILTIN_TEMPLATES:
                yield "rendered_prompt", ""
                yield "template_id", tid
                yield "templates", []
                yield "status", f"Cannot delete built-in template '{tid}'."
                yield "persona", ""
            else:
                yield "rendered_prompt", ""
                yield "template_id", tid
                yield "templates", []
                yield "status", f"Template '{tid}' not found."
                yield "persona", ""
