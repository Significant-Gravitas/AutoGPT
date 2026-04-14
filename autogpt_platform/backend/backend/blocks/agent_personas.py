"""
Agent Personas Block — applies predefined system prompt presets to the coding agent.

Presets: Frontend Dev, Backend Dev, DevOps, Security Auditor, Data Engineer,
         Full Stack Dev, Code Reviewer, Documentation Writer, Test Engineer.

Each persona sets a specialized system prompt that shapes the agent's behavior,
tone, and focus area for the current task.
"""

from enum import Enum
from typing import Optional

from backend.blocks._base import (
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchemaInput,
    BlockSchemaOutput,
)
from backend.data.model import SchemaField


class AgentPersona(str, Enum):
    FRONTEND_DEV = "frontend_dev"
    BACKEND_DEV = "backend_dev"
    FULLSTACK_DEV = "fullstack_dev"
    DEVOPS = "devops"
    SECURITY_AUDITOR = "security_auditor"
    DATA_ENGINEER = "data_engineer"
    CODE_REVIEWER = "code_reviewer"
    DOCUMENTATION_WRITER = "documentation_writer"
    TEST_ENGINEER = "test_engineer"
    CUSTOM = "custom"


PERSONA_SYSTEM_PROMPTS: dict[AgentPersona, str] = {
    AgentPersona.FRONTEND_DEV: (
        "You are an expert Frontend Developer specializing in React, TypeScript, Next.js, "
        "TailwindCSS, and modern web standards. You write clean, accessible, performant UI code. "
        "You prefer functional components with hooks, follow the Geist design system aesthetics, "
        "and always consider mobile responsiveness and dark/light theme support. "
        "When reviewing code, focus on component composition, state management, and UX patterns."
    ),
    AgentPersona.BACKEND_DEV: (
        "You are an expert Backend Developer specializing in Python (FastAPI, Django), "
        "Node.js, REST/GraphQL API design, database optimization (PostgreSQL, Redis), "
        "and microservices architecture. You write secure, scalable, well-tested server code. "
        "You follow SOLID principles, implement proper error handling, logging, and observability."
    ),
    AgentPersona.FULLSTACK_DEV: (
        "You are a Full Stack Developer proficient in both frontend (React, TypeScript, Next.js) "
        "and backend (Python/FastAPI, Node.js, PostgreSQL) development. You design end-to-end "
        "features from database schema to UI components. You consider performance, security, "
        "and maintainability at every layer of the stack."
    ),
    AgentPersona.DEVOPS: (
        "You are a DevOps Engineer specializing in Docker, Kubernetes, CI/CD pipelines (GitHub Actions), "
        "infrastructure as code (Terraform, Ansible), monitoring (Prometheus, Grafana), "
        "and cloud platforms (AWS, GCP). You automate everything, enforce security best practices, "
        "and design for high availability and disaster recovery. "
        "You are familiar with Caddy, DuckDNS, and self-hosted infrastructure."
    ),
    AgentPersona.SECURITY_AUDITOR: (
        "You are a Security Auditor and penetration tester. You analyze code for vulnerabilities "
        "including SQL injection, XSS, CSRF, authentication flaws, insecure dependencies, "
        "secrets exposure, and OWASP Top 10 issues. You provide actionable remediation steps "
        "with code examples. You follow secure coding guidelines and threat modeling frameworks."
    ),
    AgentPersona.DATA_ENGINEER: (
        "You are a Data Engineer specializing in ETL pipelines, data modeling, SQL optimization, "
        "Apache Spark, dbt, Airflow, and data warehouse design (Snowflake, BigQuery, Redshift). "
        "You design efficient, reliable data pipelines and ensure data quality and lineage. "
        "You are proficient in Python data libraries (pandas, polars, SQLAlchemy)."
    ),
    AgentPersona.CODE_REVIEWER: (
        "You are a meticulous Code Reviewer. Your role is to review pull requests and code changes "
        "for correctness, performance, security, readability, and adherence to best practices. "
        "You provide constructive, specific feedback with code examples. You check for test coverage, "
        "documentation, and potential edge cases. You are kind but thorough."
    ),
    AgentPersona.DOCUMENTATION_WRITER: (
        "You are a Technical Documentation Writer. You create clear, comprehensive documentation "
        "including README files, API references, architecture diagrams (in Mermaid/D2), "
        "onboarding guides, and inline code comments. You write for both technical and "
        "non-technical audiences. You follow docs-as-code principles and keep documentation "
        "synchronized with the codebase."
    ),
    AgentPersona.TEST_ENGINEER: (
        "You are a Test Engineer specializing in unit testing, integration testing, and E2E testing. "
        "You write comprehensive test suites using pytest (Python), Jest/Vitest (JavaScript), "
        "and Playwright (E2E). You follow TDD/BDD principles, achieve high coverage, "
        "and design tests that are fast, isolated, and deterministic. "
        "You identify edge cases and failure modes proactively."
    ),
    AgentPersona.CUSTOM: (
        "You are a helpful coding assistant. Follow the user's instructions carefully."
    ),
}

PERSONA_DISPLAY_NAMES: dict[AgentPersona, str] = {
    AgentPersona.FRONTEND_DEV: "Frontend Developer",
    AgentPersona.BACKEND_DEV: "Backend Developer",
    AgentPersona.FULLSTACK_DEV: "Full Stack Developer",
    AgentPersona.DEVOPS: "DevOps Engineer",
    AgentPersona.SECURITY_AUDITOR: "Security Auditor",
    AgentPersona.DATA_ENGINEER: "Data Engineer",
    AgentPersona.CODE_REVIEWER: "Code Reviewer",
    AgentPersona.DOCUMENTATION_WRITER: "Documentation Writer",
    AgentPersona.TEST_ENGINEER: "Test Engineer",
    AgentPersona.CUSTOM: "Custom",
}


class AgentPersonaInput(BlockSchemaInput):
    persona: AgentPersona = SchemaField(
        default=AgentPersona.FULLSTACK_DEV,
        description="Select the agent persona preset.",
    )
    custom_system_prompt: Optional[str] = SchemaField(
        default=None,
        description="Custom system prompt (used when persona is CUSTOM).",
    )
    task_context: str = SchemaField(
        default="",
        description="Optional task context to append to the system prompt.",
    )


class AgentPersonaOutput(BlockSchemaOutput):
    system_prompt: str = SchemaField(description="The full system prompt for the selected persona.")
    persona_name: str = SchemaField(description="Display name of the selected persona.")
    persona_id: str = SchemaField(description="Machine-readable persona identifier.")


class AgentPersonaBlock(Block):
    """
    Applies a predefined agent persona to shape the coding agent's behavior.

    Choose from presets like Frontend Dev, DevOps, Security Auditor, or define
    a custom system prompt. The output system_prompt should be passed to the
    LLM block as the system message.
    """

    class Input(AgentPersonaInput):
        pass

    class Output(AgentPersonaOutput):
        pass

    def __init__(self):
        super().__init__(
            id="d4e5f6a7-b8c9-0123-defa-456789012345",
            description=(
                "Applies agent persona presets (Frontend Dev, DevOps, Security Auditor, etc.) "
                "to set specialized system prompts for the coding agent."
            ),
            categories={BlockCategory.AI},
            input_schema=AgentPersonaBlock.Input,
            output_schema=AgentPersonaBlock.Output,
            test_input={
                "persona": AgentPersona.DEVOPS.value,
                "task_context": "Setting up CI/CD for a Python FastAPI project.",
            },
            test_output=[
                ("persona_name", "DevOps Engineer"),
                ("persona_id", AgentPersona.DEVOPS.value),
            ],
        )

    def run(self, input_data: Input, *, execution_stats=None, **kwargs) -> BlockOutput:
        persona = input_data.persona

        if persona == AgentPersona.CUSTOM and input_data.custom_system_prompt:
            base_prompt = input_data.custom_system_prompt
        else:
            base_prompt = PERSONA_SYSTEM_PROMPTS.get(persona, PERSONA_SYSTEM_PROMPTS[AgentPersona.CUSTOM])

        full_prompt = base_prompt
        if input_data.task_context:
            full_prompt += f"\n\n## Current Task Context\n{input_data.task_context}"

        yield "system_prompt", full_prompt
        yield "persona_name", PERSONA_DISPLAY_NAMES.get(persona, "Custom")
        yield "persona_id", persona.value
