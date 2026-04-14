"""
Screenshot to Task Block — converts a screenshot or image into a structured
coding task using vision LLM analysis.

Use cases:
- Screenshot of a UI bug → "Fix the alignment issue in the header component"
- Screenshot of a design mockup → "Implement this UI in React/TailwindCSS"
- Screenshot of an error message → "Debug and fix this error"
- Screenshot of a feature request → "Implement this feature"
"""

import base64
import logging
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


class ScreenshotIntent(str, Enum):
    UI_BUG = "ui_bug"
    UI_IMPLEMENTATION = "ui_implementation"
    ERROR_FIX = "error_fix"
    FEATURE_REQUEST = "feature_request"
    CODE_REVIEW = "code_review"
    AUTO_DETECT = "auto_detect"


INTENT_PROMPTS = {
    ScreenshotIntent.UI_BUG: (
        "This screenshot shows a UI bug or visual issue. "
        "Describe the bug precisely and generate a coding task to fix it. "
        "Include: what component is affected, what the issue is, and what the expected behavior should be."
    ),
    ScreenshotIntent.UI_IMPLEMENTATION: (
        "This screenshot shows a UI design or mockup. "
        "Generate a detailed coding task to implement this UI. "
        "Include: component structure, styling details, responsive behavior, and interactions."
    ),
    ScreenshotIntent.ERROR_FIX: (
        "This screenshot shows an error message or stack trace. "
        "Analyze the error and generate a debugging/fix task. "
        "Include: error type, likely cause, and suggested investigation steps."
    ),
    ScreenshotIntent.FEATURE_REQUEST: (
        "This screenshot shows a feature or functionality to implement. "
        "Generate a detailed feature implementation task. "
        "Include: what needs to be built, user interactions, and technical requirements."
    ),
    ScreenshotIntent.CODE_REVIEW: (
        "This screenshot shows code. "
        "Generate a code review task identifying issues and improvements. "
        "Include: potential bugs, style issues, performance concerns, and security considerations."
    ),
    ScreenshotIntent.AUTO_DETECT: (
        "Analyze this screenshot and determine what type of coding task it represents. "
        "It could be a UI bug, design mockup, error message, feature request, or code snippet. "
        "Generate a precise, actionable coding task based on what you see."
    ),
}


class ScreenshotToTaskInput(BlockSchemaInput):
    image_path: str = SchemaField(
        default="",
        description="Local file path to the screenshot/image.",
    )
    image_url: str = SchemaField(
        default="",
        description="URL of the screenshot/image (used if image_path is not provided).",
    )
    image_base64: str = SchemaField(
        default="",
        description="Base64-encoded image data (used if neither path nor URL is provided).",
    )
    intent: ScreenshotIntent = SchemaField(
        default=ScreenshotIntent.AUTO_DETECT,
        description="Intent of the screenshot: ui_bug, ui_implementation, error_fix, feature_request, code_review, or auto_detect.",
    )
    additional_context: str = SchemaField(
        default="",
        description="Additional context about the screenshot (e.g., 'This is from the React frontend, file: Header.tsx').",
    )
    tech_stack: list = SchemaField(
        default_factory=list,
        description="Tech stack context (e.g., ['React', 'TypeScript', 'TailwindCSS']).",
    )
    persona: str = SchemaField(
        default="frontend_dev",
        description="Agent persona to use for the generated task.",
    )
    model: str = SchemaField(
        default="gpt-4.1-mini",
        description="Vision-capable LLM model to use for image analysis.",
    )
    api_key: str = SchemaField(
        default="",
        description="OpenAI API key (uses environment variable if not provided).",
    )


class ScreenshotToTaskOutput(BlockSchemaOutput):
    task_prompt: str = SchemaField(description="Generated coding task prompt.")
    task_title: str = SchemaField(description="Short title for the task.")
    detected_intent: str = SchemaField(description="Detected or specified intent.")
    persona: str = SchemaField(description="Recommended persona for the task.")
    image_description: str = SchemaField(description="Description of what the image shows.")
    status: str = SchemaField(description="Operation status.")


class ScreenshotToTaskBlock(Block):
    """
    Converts a screenshot into a structured coding task using vision AI.

    Drag-and-drop a screenshot of a UI bug, design mockup, error message,
    or feature request. The vision LLM analyzes the image and generates
    a precise, actionable task prompt for the coding agent.
    """

    class Input(ScreenshotToTaskInput):
        pass

    class Output(ScreenshotToTaskOutput):
        pass

    def __init__(self):
        super().__init__(
            id="a3b4c5d6-e7f8-9012-abcd-345678901234",
            description=(
                "Converts screenshots to coding tasks using vision AI. "
                "Supports UI bugs, design mockups, error messages, and feature requests."
            ),
            categories={BlockCategory.AI, BlockCategory.DEVELOPER_TOOLS},
            input_schema=ScreenshotToTaskBlock.Input,
            output_schema=ScreenshotToTaskBlock.Output,
            test_input={
                "image_path": "",
                "intent": ScreenshotIntent.AUTO_DETECT.value,
                "additional_context": "Test context",
            },
            test_output=[
                ("status", "No image provided. Please provide image_path, image_url, or image_base64."),
            ],
        )

    def run(self, input_data: Input, *, execution_stats=None, **kwargs) -> BlockOutput:
        # Validate image input
        if not input_data.image_path and not input_data.image_url and not input_data.image_base64:
            yield "task_prompt", ""
            yield "task_title", ""
            yield "detected_intent", input_data.intent.value
            yield "persona", input_data.persona
            yield "image_description", ""
            yield "status", "No image provided. Please provide image_path, image_url, or image_base64."
            return

        # Build image content for API
        image_content = None
        if input_data.image_path:
            try:
                img_path = Path(input_data.image_path)
                if not img_path.exists():
                    yield "task_prompt", ""
                    yield "task_title", ""
                    yield "detected_intent", input_data.intent.value
                    yield "persona", input_data.persona
                    yield "image_description", ""
                    yield "status", f"Image file not found: {input_data.image_path}"
                    return
                img_data = base64.b64encode(img_path.read_bytes()).decode()
                suffix = img_path.suffix.lower().lstrip(".")
                media_type = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}.get(suffix, "png")
                image_content = {"type": "base64", "data": img_data, "media_type": f"image/{media_type}"}
            except Exception as e:
                yield "task_prompt", ""
                yield "task_title", ""
                yield "detected_intent", input_data.intent.value
                yield "persona", input_data.persona
                yield "image_description", ""
                yield "status", f"Failed to read image: {e}"
                return
        elif input_data.image_url:
            image_content = {"type": "url", "url": input_data.image_url}
        elif input_data.image_base64:
            image_content = {"type": "base64", "data": input_data.image_base64, "media_type": "image/png"}

        # Build prompt
        intent_instruction = INTENT_PROMPTS.get(input_data.intent, INTENT_PROMPTS[ScreenshotIntent.AUTO_DETECT])
        tech_context = f"\nTech Stack: {', '.join(input_data.tech_stack)}" if input_data.tech_stack else ""
        extra_context = f"\nAdditional Context: {input_data.additional_context}" if input_data.additional_context else ""

        system_prompt = (
            "You are an expert software engineer analyzing screenshots to generate precise coding tasks. "
            "Always respond with a JSON object containing: "
            "'title' (short task title), 'description' (what the image shows), "
            "'task_prompt' (detailed coding task), 'intent' (detected intent), 'persona' (recommended agent persona)."
        )

        user_prompt = (
            f"{intent_instruction}{tech_context}{extra_context}\n\n"
            "Respond with a JSON object with keys: title, description, task_prompt, intent, persona."
        )

        # Call vision LLM
        try:
            import os
            from openai import OpenAI

            api_key = input_data.api_key or os.environ.get("OPENAI_API_KEY", "")
            client = OpenAI(api_key=api_key) if api_key else OpenAI()

            # Build messages
            if image_content["type"] == "url":
                img_msg = {"type": "image_url", "image_url": {"url": image_content["url"]}}
            else:
                img_msg = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{image_content['media_type']};base64,{image_content['data']}"
                    },
                }

            response = client.chat.completions.create(
                model=input_data.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        img_msg,
                        {"type": "text", "text": user_prompt},
                    ]},
                ],
                max_tokens=1500,
                response_format={"type": "json_object"},
            )

            import json
            result = json.loads(response.choices[0].message.content)
            yield "task_prompt", result.get("task_prompt", "")
            yield "task_title", result.get("title", "Screenshot Task")
            yield "detected_intent", result.get("intent", input_data.intent.value)
            yield "persona", result.get("persona", input_data.persona)
            yield "image_description", result.get("description", "")
            yield "status", "Screenshot analyzed successfully."

        except ImportError:
            yield "task_prompt", ""
            yield "task_title", ""
            yield "detected_intent", input_data.intent.value
            yield "persona", input_data.persona
            yield "image_description", ""
            yield "status", "openai package not installed. Run: pip install openai"
        except Exception as e:
            yield "task_prompt", ""
            yield "task_title", ""
            yield "detected_intent", input_data.intent.value
            yield "persona", input_data.persona
            yield "image_description", ""
            yield "status", f"Vision analysis failed: {e}"
