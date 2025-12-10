"""
Long-Running Initializer Block.

This block sets up the initial environment for a long-running agent project.
It is run ONCE at the start of a new project and creates:
- Feature list file with all features marked as pending
- Progress log file
- Init script for environment setup
- Initial git commit

Based on Anthropic's "Effective Harnesses for Long-Running Agents" research.
"""

import json
import logging
import uuid
from typing import Optional

from pydantic import SecretStr

from backend.blocks.llm import AIStructuredResponseGeneratorBlock, LlmModel
from backend.data.block import Block, BlockCategory, BlockOutput, BlockType
from backend.data.model import (
    APIKeyCredentials,
    CredentialsField,
    CredentialsMetaInput,
    SchemaField,
)
from backend.integrations.providers import ProviderName

from .models import (
    FeatureCategory,
    FeatureListItem,
    FeatureStatus,
    InitializerConfig,
    ProgressEntry,
    ProgressEntryType,
    SessionStatus,
)
from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class LongRunningInitializerBlock(Block):
    """
    Initialize a long-running agent project.

    This block sets up the environment for a project that will span
    multiple agent sessions. It creates:
    - A feature list based on the project specification
    - A progress log for tracking work across sessions
    - An init.sh script for environment setup
    - An initial git repository with the setup

    Use this block ONLY for the first session of a new project.
    """

    class Input(CredentialsMetaInput):
        project_name: str = SchemaField(
            description="Name of the project to create"
        )
        project_description: str = SchemaField(
            description="Detailed description of what the project should accomplish. "
            "Include all features, requirements, and acceptance criteria."
        )
        working_directory: str = SchemaField(
            description="Directory where the project will be created"
        )
        generate_features_with_ai: bool = SchemaField(
            default=True,
            description="Use AI to generate a comprehensive feature list from the description",
        )
        custom_features: list[dict] = SchemaField(
            default=[],
            description="Custom features to add (each dict should have 'description' and optionally 'category', 'steps', 'priority')",
        )
        init_commands: list[str] = SchemaField(
            default=[],
            description="Shell commands to include in the init.sh script",
        )
        initialize_git: bool = SchemaField(
            default=True,
            description="Whether to initialize a git repository",
        )
        credentials: CredentialsMetaInput[
            ProviderName.OPENAI, ProviderName.ANTHROPIC, ProviderName.GROQ
        ] = CredentialsField(
            description="LLM API credentials for feature generation (if using AI)",
            required=False,
        )

    class Output(Block.Output):
        session_id: str = SchemaField(
            description="Unique identifier for this long-running session"
        )
        feature_count: int = SchemaField(
            description="Number of features generated"
        )
        feature_list_path: str = SchemaField(
            description="Path to the feature list file"
        )
        progress_log_path: str = SchemaField(
            description="Path to the progress log file"
        )
        init_script_path: str = SchemaField(
            description="Path to the init script"
        )
        status: str = SchemaField(
            description="Status of the initialization"
        )
        error: str = SchemaField(
            description="Error message if initialization failed",
            default="",
        )

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            description="Initialize a long-running agent project with feature list, progress tracking, and environment setup",
            input_schema=LongRunningInitializerBlock.Input,
            output_schema=LongRunningInitializerBlock.Output,
            categories={BlockCategory.AGENT},
            block_type=BlockType.AGENT,
        )

    async def run(
        self,
        input_data: Input,
        *,
        credentials: Optional[APIKeyCredentials] = None,
        **kwargs,
    ) -> BlockOutput:
        session_id = str(uuid.uuid4())[:8]

        try:
            # Initialize session manager
            manager = SessionManager(input_data.working_directory)

            # Check if project already exists
            existing_state = manager.load_session_state()
            if existing_state and existing_state.status != SessionStatus.INITIALIZING:
                yield "error", f"Project already initialized at {input_data.working_directory}"
                yield "status", "failed"
                return

            # Create session state
            state = manager.create_session_state(
                project_name=input_data.project_name,
                project_description=input_data.project_description,
            )

            # Start progress log
            manager.start_session_log(session_id)

            # Generate features
            features = []

            if input_data.generate_features_with_ai and credentials:
                ai_features = await self._generate_features_with_ai(
                    input_data.project_name,
                    input_data.project_description,
                    credentials,
                )
                features.extend(ai_features)

            # Add custom features
            for custom in input_data.custom_features:
                if "description" in custom:
                    features.append(custom)

            # Create feature list
            if features:
                manager.create_feature_list(
                    project_name=input_data.project_name,
                    project_description=input_data.project_description,
                    features=features,
                )

            # Create init script
            init_commands = input_data.init_commands or [
                "echo 'Project initialized successfully'",
            ]
            manager.create_init_script(
                commands=init_commands,
                description=f"Setup script for {input_data.project_name}",
            )

            # Initialize git if requested
            if input_data.initialize_git:
                manager.initialize_git()

            # Update session state to ready
            manager.update_session_status(SessionStatus.READY, session_id)

            # Log completion
            manager.add_progress_entry(
                ProgressEntry(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    entry_type=ProgressEntryType.ENVIRONMENT_SETUP,
                    title="Project initialization complete",
                    description=f"Created {len(features)} features, init script, and git repository",
                )
            )

            yield "session_id", session_id
            yield "feature_count", len(features)
            yield "feature_list_path", str(manager.feature_list_path)
            yield "progress_log_path", str(manager.progress_log_path)
            yield "init_script_path", str(manager.init_script_path)
            yield "status", "success"

        except Exception as e:
            logger.exception(f"Failed to initialize long-running project: {e}")
            yield "error", str(e)
            yield "status", "failed"

    async def _generate_features_with_ai(
        self,
        project_name: str,
        project_description: str,
        credentials: APIKeyCredentials,
    ) -> list[dict]:
        """Generate a comprehensive feature list using AI."""

        system_prompt = """You are an expert software architect generating a comprehensive feature list for a project.

Your task is to analyze the project description and create a detailed list of features that need to be implemented.

For each feature:
1. Write a clear, testable description of the feature
2. Assign a category: functional, ui, integration, performance, security, documentation, testing, infrastructure
3. Provide verification steps (how to test the feature works)
4. Assign a priority (1=highest, 10=lowest)

Important guidelines:
- Break down complex features into smaller, independently testable features
- Each feature should be completable in a single session
- Include edge cases and error handling as separate features
- Be comprehensive - include features that might be implicit in the description
- Think about the user journey end-to-end"""

        user_prompt = f"""Project Name: {project_name}

Project Description:
{project_description}

Generate a comprehensive list of features for this project. Output as JSON array with the following structure:
[
  {{
    "id": "feature_001",
    "category": "functional",
    "description": "Clear description of what the feature does",
    "steps": ["Step 1 to verify", "Step 2 to verify"],
    "priority": 1
  }}
]

Generate at least 20 features covering all aspects of the project."""

        try:
            # Determine model based on provider
            if credentials.provider == "anthropic":
                model = LlmModel.CLAUDE_3_5_SONNET
            elif credentials.provider == "openai":
                model = LlmModel.GPT4O
            else:
                model = LlmModel.LLAMA3_70B

            structured_block = AIStructuredResponseGeneratorBlock()

            structured_input = AIStructuredResponseGeneratorBlock.Input(
                prompt=user_prompt,
                sys_prompt=system_prompt,
                expected_format={
                    "features": [
                        {
                            "id": "string",
                            "category": "string",
                            "description": "string",
                            "steps": ["string"],
                            "priority": 1,
                        }
                    ]
                },
                model=model,
                credentials={
                    "provider": credentials.provider,
                    "id": credentials.id,
                    "type": credentials.type,
                    "title": credentials.title,
                },
                max_tokens=4000,
                retry=3,
            )

            async for output_name, output_data in structured_block.run(
                structured_input, credentials=credentials
            ):
                if output_name == "response":
                    return output_data.get("features", [])

            return []

        except Exception as e:
            logger.error(f"Failed to generate features with AI: {e}")
            # Return some default features if AI generation fails
            return [
                {
                    "id": "feature_001",
                    "category": "functional",
                    "description": "Basic project setup and structure",
                    "steps": ["Verify project files exist", "Verify dependencies installed"],
                    "priority": 1,
                },
                {
                    "id": "feature_002",
                    "category": "functional",
                    "description": "Core functionality as described in project spec",
                    "steps": ["Verify main features work end-to-end"],
                    "priority": 2,
                },
            ]
