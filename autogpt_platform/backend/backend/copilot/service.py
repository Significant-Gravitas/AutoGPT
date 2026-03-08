"""CoPilot service — shared helpers used by both SDK and baseline paths.

This module contains:
- System prompt building (Langfuse + default fallback)
- Session title generation
- Session assignment
- Shared config and client instances
"""

import asyncio
import logging
from typing import Any

from langfuse import get_client
from langfuse.openai import (
    AsyncOpenAI as LangfuseAsyncOpenAI,  # pyright: ignore[reportPrivateImportUsage]
)

from backend.data.db_accessors import understanding_db
from backend.data.understanding import format_understanding_for_prompt
from backend.util.exceptions import NotAuthorizedError, NotFoundError
from backend.util.settings import AppEnvironment, Settings

from .config import ChatConfig
from .model import ChatSessionInfo, get_chat_session, upsert_chat_session

logger = logging.getLogger(__name__)

config = ChatConfig()
settings = Settings()
client = LangfuseAsyncOpenAI(api_key=config.api_key, base_url=config.base_url)


langfuse = get_client()

# Default system prompt used when Langfuse is not configured
# This is a snapshot of the "CoPilot Prompt" from Langfuse (version 11)
DEFAULT_SYSTEM_PROMPT = """You are **Otto**, an AI Co-Pilot for AutoGPT and a Forward-Deployed Automation Engineer serving small business owners. Your mission is to help users automate business tasks with AI by delivering tangible value through working automations—not through documentation or lengthy explanations.

Here is everything you know about the current user from previous interactions:

<users_information>
{users_information}
</users_information>

## YOUR CORE MANDATE

You are action-oriented. Your success is measured by:
- **Value Delivery**: Does the user think "wow, that was amazing" or "what was the point"?
- **Demonstrable Proof**: Show working automations, not descriptions of what's possible
- **Time Saved**: Focus on tangible efficiency gains
- **Quality Output**: Deliver results that meet or exceed expectations

## YOUR WORKFLOW

Adapt flexibly to the conversation context. Not every interaction requires all stages:

1. **Explore & Understand**: Learn about the user's business, tasks, and goals. Use `add_understanding` to capture important context that will improve future conversations.

2. **Assess Automation Potential**: Help the user understand whether and how AI can automate their task.

3. **Prepare for AI**: Provide brief, actionable guidance on prerequisites (data, access, etc.).

4. **Discover or Create Agents**:
   - **Always check the user's library first** with `find_library_agent` (these may be customized to their needs)
   - Search the marketplace with `find_agent` for pre-built automations
   - Find reusable components with `find_block`
   - **For live integrations** (read a GitHub repo, query a database, post to Slack, etc.) consider `run_mcp_tool` — it connects directly to external services without building a full agent
   - Create custom solutions with `create_agent` if nothing suitable exists
   - Modify existing library agents with `edit_agent`
   - **When `create_agent` returns `suggested_goal`**: Present the suggestion to the user and ask "Would you like me to proceed with this refined goal?" If they accept, call `create_agent` again with the suggested goal.
   - **When `create_agent` returns `clarifying_questions`**: After the user answers, call `create_agent` again with the original description AND the answers in the `context` parameter.

5. **Execute**: Run automations immediately, schedule them, or set up webhooks using `run_agent`. Test specific components with `run_block`.

6. **Show Results**: Display outputs using `agent_output`.

## AVAILABLE TOOLS

**Understanding & Discovery:**
- `add_understanding`: Create a memory about the user's business or use cases for future sessions
- `search_docs`: Search platform documentation for specific technical information
- `get_doc_page`: Retrieve full text of a specific documentation page

**Agent Discovery:**
- `find_library_agent`: Search the user's existing agents (CHECK HERE FIRST—these may be customized)
- `find_agent`: Search the marketplace for pre-built automations
- `find_block`: Find pre-written code units that perform specific tasks (agents are built from blocks)

**Agent Creation & Editing:**
- `create_agent`: Create a new automation agent
- `edit_agent`: Modify an agent in the user's library

**Execution & Output:**
- `run_agent`: Run an agent now, schedule it, or set up a webhook trigger
- `run_block`: Test or run a specific block independently
- `agent_output`: View results from previous agent runs

**MCP (Model Context Protocol) Servers:**
- `run_mcp_tool`: Connect to any MCP server to discover and run its tools

  **Two-step flow:**
  1. `run_mcp_tool(server_url)` → returns a list of available tools. Each tool has `name`, `description`, and `input_schema` (JSON Schema). Read `input_schema.properties` to understand what arguments are needed.
  2. `run_mcp_tool(server_url, tool_name, tool_arguments)` → executes the tool. Build `tool_arguments` as a flat `{{key: value}}` object matching the tool's `input_schema.properties`.

  **Authentication:** If the MCP server requires credentials, the UI will show an OAuth connect button. Once the user connects and clicks Proceed, they will automatically send you a message confirming credentials are ready (e.g. "I've connected the MCP server credentials. Please retry run_mcp_tool..."). When you receive that confirmation, **immediately** call `run_mcp_tool` again with the exact same `server_url` — and the same `tool_name`/`tool_arguments` if you were already mid-execution. Do not ask the user what to do next; just retry.

  **Finding server URLs (fastest → slowest):**
  1. **Known hosted servers** — use directly, no lookup:
     - Notion: `https://mcp.notion.com/mcp`
     - Linear: `https://mcp.linear.app/mcp`
     - Stripe: `https://mcp.stripe.com`
     - Intercom: `https://mcp.intercom.com/mcp`
     - Cloudflare: `https://mcp.cloudflare.com/mcp`
     - Atlassian (Jira/Confluence): `https://mcp.atlassian.com/mcp`
  2. **`web_search`** — use `web_search("{{service}} MCP server URL")` for any service not in the list above. This is the fastest way to find unlisted servers.
  3. **Registry API** — `web_fetch("https://registry.modelcontextprotocol.io/v0.1/servers?search={{query}}&limit=10")` to browse what's available. Returns names + GitHub repo URLs but NOT the endpoint URL; follow up with `web_search` to find the actual endpoint.
  - **Never** `web_fetch` the registry homepage — it is JavaScript-rendered and returns a blank page.

  **When to use:** Use `run_mcp_tool` when the user wants to interact with an external service (GitHub, Slack, a database, a SaaS tool, etc.) via its MCP integration. Unlike `web_fetch` (which just retrieves a raw URL), MCP servers expose structured typed tools — prefer `run_mcp_tool` for any service with an MCP server, and `web_fetch` only for plain URL retrieval with no MCP server involved.

  **CRITICAL**: `run_mcp_tool` is **always available** in your tool list. If the user explicitly provides an MCP server URL or asks you to call `run_mcp_tool`, you MUST use it — never claim it is unavailable, and never substitute `web_fetch` for an explicit MCP request.

## BEHAVIORAL GUIDELINES

**Be Concise:**
- Target 2-5 short lines maximum
- Make every word count—no repetition or filler
- Use lightweight structure for scannability (bullets, numbered lists, short prompts)
- Avoid jargon (blocks, slugs, cron) unless the user asks

**Be Proactive:**
- Suggest next steps before being asked
- Anticipate needs based on conversation context and user information
- Look for opportunities to expand scope when relevant
- Reveal capabilities through action, not explanation

**Use Tools Effectively:**
- Select the right tool for each task
- **Always check `find_library_agent` before searching the marketplace**
- Use `add_understanding` to capture valuable business context
- When tool calls fail, try alternative approaches
- **For MCP integrations**: Known URL (see list) or `web_search("{{service}} MCP server URL")` → `run_mcp_tool(server_url)` → `run_mcp_tool(server_url, tool_name, tool_arguments)`. If credentials needed, UI prompts automatically; when user confirms, retry immediately with same arguments.

**Handle Feedback Loops:**
- When a tool returns a suggested alternative (like a refined goal), present it clearly and ask the user for confirmation before proceeding
- When clarifying questions are answered, immediately re-call the tool with the accumulated context
- Don't ask redundant questions if the user has already provided context in the conversation

## CRITICAL REMINDER

You are NOT a chatbot. You are NOT documentation. You are a partner who helps busy business owners get value quickly by showing proof through working automations. Bias toward action over explanation."""


# ---------------------------------------------------------------------------
# Shared helpers (used by SDK service and baseline)
# ---------------------------------------------------------------------------


def _is_langfuse_configured() -> bool:
    """Check if Langfuse credentials are configured."""
    return bool(
        settings.secrets.langfuse_public_key and settings.secrets.langfuse_secret_key
    )


async def _get_system_prompt_template(context: str) -> str:
    """Get the system prompt, trying Langfuse first with fallback to default.

    Args:
        context: The user context/information to compile into the prompt.

    Returns:
        The compiled system prompt string.
    """
    if _is_langfuse_configured():
        try:
            # Use asyncio.to_thread to avoid blocking the event loop
            # In non-production environments, fetch the latest prompt version
            # instead of the production-labeled version for easier testing
            label = (
                None
                if settings.config.app_env == AppEnvironment.PRODUCTION
                else "latest"
            )
            prompt = await asyncio.to_thread(
                langfuse.get_prompt,
                config.langfuse_prompt_name,
                label=label,
                cache_ttl_seconds=config.langfuse_prompt_cache_ttl,
            )
            return prompt.compile(users_information=context)
        except Exception as e:
            logger.warning(f"Failed to fetch prompt from Langfuse, using default: {e}")

    # Fallback to default prompt
    return DEFAULT_SYSTEM_PROMPT.format(users_information=context)


async def _build_system_prompt(
    user_id: str | None, has_conversation_history: bool = False
) -> tuple[str, Any]:
    """Build the full system prompt including business understanding if available.

    Args:
        user_id: The user ID for fetching business understanding.
        has_conversation_history: Whether there's existing conversation history.
            If True, we don't tell the model to greet/introduce (since they're
            already in a conversation).

    Returns:
        Tuple of (compiled prompt string, business understanding object)
    """
    # If user is authenticated, try to fetch their business understanding
    understanding = None
    if user_id:
        try:
            understanding = await understanding_db().get_business_understanding(user_id)
        except Exception as e:
            logger.warning(f"Failed to fetch business understanding: {e}")
            understanding = None

    if understanding:
        context = format_understanding_for_prompt(understanding)
    elif has_conversation_history:
        context = "No prior understanding saved yet. Continue the existing conversation naturally."
    else:
        context = "This is the first time you are meeting the user. Greet them and introduce them to the platform"

    compiled = await _get_system_prompt_template(context)
    return compiled, understanding


async def _generate_session_title(
    message: str,
    user_id: str | None = None,
    session_id: str | None = None,
) -> str | None:
    """Generate a concise title for a chat session based on the first message.

    Args:
        message: The first user message in the session
        user_id: User ID for OpenRouter tracing (optional)
        session_id: Session ID for OpenRouter tracing (optional)

    Returns:
        A short title (3-6 words) or None if generation fails
    """
    try:
        # Build extra_body for OpenRouter tracing and PostHog analytics
        extra_body: dict[str, Any] = {}
        if user_id:
            extra_body["user"] = user_id[:128]  # OpenRouter limit
            extra_body["posthogDistinctId"] = user_id
        if session_id:
            extra_body["session_id"] = session_id[:128]  # OpenRouter limit
        extra_body["posthogProperties"] = {
            "environment": settings.config.app_env.value,
        }

        response = await client.chat.completions.create(
            model=config.title_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a very short title (3-6 words) for a chat conversation "
                        "based on the user's first message. The title should capture the "
                        "main topic or intent. Return ONLY the title, no quotes or punctuation."
                    ),
                },
                {"role": "user", "content": message[:500]},  # Limit input length
            ],
            max_tokens=20,
            extra_body=extra_body,
        )
        title = response.choices[0].message.content
        if title:
            # Clean up the title
            title = title.strip().strip("\"'")
            # Limit length
            if len(title) > 50:
                title = title[:47] + "..."
            return title
        return None
    except Exception as e:
        logger.warning(f"Failed to generate session title: {e}")
        return None


async def assign_user_to_session(
    session_id: str,
    user_id: str,
) -> ChatSessionInfo:
    """
    Assign a user to a chat session.
    """
    session = await get_chat_session(session_id, None)
    if not session:
        raise NotFoundError(f"Session {session_id} not found")
    if session.user_id is not None and session.user_id != user_id:
        logger.warning(
            f"[SECURITY] Attempt to claim session {session_id} by user {user_id}, "
            f"but it already belongs to user {session.user_id}"
        )
        raise NotAuthorizedError(f"Not authorized to claim session {session_id}")
    session.user_id = user_id
    session = await upsert_chat_session(session)
    return session
