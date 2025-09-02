"""Chat streaming service for handling OpenAI chat completions with tool calling."""

import asyncio
import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

try:
    from openai import AsyncOpenAI
except ImportError:
    # Fallback for older OpenAI versions
    from openai import OpenAI as AsyncOpenAI  # type: ignore

from prisma.enums import ChatMessageRole

from backend.server.v2.chat import db
from backend.server.v2.chat.tools import tools

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ChatStreamingService:
    """Service for streaming chat responses with tool calling support."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the chat streaming service.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY env var.
        """
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    async def stream_chat_response(
        self,
        session_id: str,
        user_message: str,
        user_id: str,
        model: str = "gpt-4o",
        max_messages: int = 50,
    ) -> AsyncGenerator[str, None]:
        """Stream OpenAI chat response with tool calling support.

        This generator handles:
        1. Streaming text responses word by word
        2. Tool call detection and execution
        3. UI element streaming for tool interactions
        4. Persisting messages to database

        Args:
            session_id: Chat session ID
            user_message: User's input message
            user_id: User ID for authentication
            model: OpenAI model to use
            max_messages: Maximum context messages to include

        Yields:
            SSE formatted JSON strings with either:
            - {"type": "text", "content": "..."}  for text chunks
            - {"type": "html", "content": "..."}  for UI elements
            - {"type": "error", "content": "..."}  for errors
        """
        try:
            # Store user message in database
            await db.create_chat_message(
                session_id=session_id,
                content=user_message,
                role=ChatMessageRole.USER,
            )

            # Get conversation context
            context = await db.get_conversation_context(
                session_id=session_id, max_messages=max_messages, include_system=True
            )

            # Add comprehensive system prompt if this is the first message
            if not any(msg.get("role") == "system" for msg in context):
                system_prompt = """# AutoGPT Agent Setup Assistant

You are a helpful AI assistant specialized in helping users discover and set up AutoGPT agents that solve their specific business problems. Your primary goal is to deliver immediate value by getting users set up with the right agents quickly and efficiently.

## Your Core Responsibilities:

### 1. UNDERSTAND THE USER'S PROBLEM
- Ask targeted questions to understand their specific business challenge
- Identify their industry, pain points, and desired outcomes
- Determine their technical comfort level and available resources

### 2. DISCOVER SUITABLE AGENTS
- Use the `find_agent` tool to search the AutoGPT marketplace for relevant agents
- Look for agents that directly address their stated problem
- Consider both specialized agents and general-purpose tools that could help
- Present 2-3 agent options with brief descriptions

### 3. VALIDATE AGENT FIT
- Explain how each recommended agent addresses their specific problem
- Ask if the recommended agents align with their needs
- Be prepared to search again with different keywords if needed
- Focus on agents that provide immediate, measurable value

### 4. GET AGENT DETAILS
- Once user shows interest in an agent, use `get_agent_details` to get comprehensive information
- This will include credential requirements, input specifications, and setup instructions
- Pay special attention to authentication requirements

### 5. HANDLE AUTHENTICATION
- If `get_agent_details` returns an authentication error, clearly explain that sign-in is required
- Guide users through the login process
- Reassure them that this is necessary for security and personalization
- After successful login, proceed with agent details

### 6. UNDERSTAND CREDENTIAL REQUIREMENTS
- Review the detailed agent information for credential needs
- Explain what each credential is used for
- Guide users on where to obtain required credentials
- Be prepared to help them through the credential setup process

### 7. SET UP THE AGENT
- Use the `setup_agent` tool to configure the agent for the user
- Set appropriate schedules, inputs, and credentials
- Choose webhook vs scheduled execution based on user preference
- Ensure all required credentials are properly configured

### 8. COMPLETE THE SETUP
- Confirm successful agent setup
- Provide clear next steps for using the agent
- Direct users to view their newly set up agent
- Offer assistance with any follow-up questions

## Important Guidelines:

### CONVERSATION FLOW:
- Keep responses conversational and friendly
- Ask one question at a time to avoid overwhelming users
- Use the available tools proactively to gather information
- Always move the conversation forward toward setup completion

### AUTHENTICATION HANDLING:
- Be transparent about why authentication is needed
- Explain that it's for security and personalization
- Reassure users that their data is safe
- Guide them smoothly through the process

### AGENT SELECTION:
- Focus on agents that solve the user's immediate problem
- Consider both simple and advanced options
- Explain the trade-offs between different agents
- Prioritize agents with clear, immediate value

### TECHNICAL EXPLANATIONS:
- Explain technical concepts in simple, business-friendly terms
- Avoid jargon unless explaining it
- Focus on benefits and outcomes rather than technical details
- Be patient and thorough in explanations

### ERROR HANDLING:
- If a tool fails, explain what happened and try alternatives
- If authentication fails, guide users through troubleshooting
- If agent setup fails, identify the issue and help resolve it
- Always provide clear next steps

## Your Success Metrics:
- Users successfully identify agents that solve their problems
- Users complete the authentication process
- Users have agents set up and running
- Users understand how to use their new agents
- Users feel confident and satisfied with the setup process

Remember: Your goal is to deliver immediate value by getting users set up with AutoGPT agents that solve their real business problems. Be proactive, helpful, and focused on successful outcomes."""

                context.insert(0, {"role": "system", "content": system_prompt})

            # Add current user message to context
            context.append({"role": "user", "content": user_message})

            logger.info(f"Starting chat stream for session {session_id}")

            # Loop to handle tool calls and continue conversation
            while True:
                try:
                    logger.info("Creating OpenAI chat completion stream...")

                    # Create the stream
                    stream = await self.client.chat.completions.create(
                        model=model,
                        messages=context,
                        tools=tools,
                        tool_choice="auto",
                        stream=True,
                    )

                    # Variables to accumulate the response
                    assistant_message: str = ""
                    tool_calls: List[Dict[str, Any]] = []
                    finish_reason: Optional[str] = None

                    # Process the stream
                    async for chunk in stream:
                        if chunk.choices:
                            choice = chunk.choices[0]
                            delta = choice.delta

                            # Capture finish reason
                            if choice.finish_reason:
                                finish_reason = choice.finish_reason
                                logger.info(f"Finish reason: {finish_reason}")

                            # Handle content streaming
                            if delta.content:
                                assistant_message += delta.content
                                # Stream word by word for nice effect
                                words = delta.content.split(" ")
                                for word in words:
                                    if word:
                                        yield f"data: {json.dumps({'type': 'text', 'content': word + ' '})}\n\n"
                                        await asyncio.sleep(0.02)

                            # Handle tool calls
                            if delta.tool_calls:
                                for tc_chunk in delta.tool_calls:
                                    idx = tc_chunk.index

                                    # Ensure we have a tool call object at this index
                                    while len(tool_calls) <= idx:
                                        tool_calls.append(
                                            {
                                                "id": "",
                                                "type": "function",
                                                "function": {
                                                    "name": "",
                                                    "arguments": "",
                                                },
                                            }
                                        )

                                    # Accumulate the tool call data
                                    if tc_chunk.id:
                                        tool_calls[idx]["id"] = tc_chunk.id
                                    if tc_chunk.function:
                                        if tc_chunk.function.name:
                                            tool_calls[idx]["function"][
                                                "name"
                                            ] = tc_chunk.function.name
                                        if tc_chunk.function.arguments:
                                            tool_calls[idx]["function"][
                                                "arguments"
                                            ] += tc_chunk.function.arguments

                    logger.info(f"Stream complete. Finish reason: {finish_reason}")

                    # Save assistant message to database if there was content
                    if assistant_message or tool_calls:
                        await db.create_chat_message(
                            session_id=session_id,
                            content=assistant_message if assistant_message else "",
                            role=ChatMessageRole.ASSISTANT,
                            tool_calls=tool_calls if tool_calls else None,
                        )

                    # Check if we need to execute tools
                    if finish_reason == "tool_calls" and tool_calls:
                        logger.info(f"Processing {len(tool_calls)} tool call(s)")

                        # Add assistant message with tool calls to context
                        context.append(
                            {
                                "role": "assistant",
                                "content": (
                                    assistant_message if assistant_message else None
                                ),
                                "tool_calls": tool_calls,
                            }
                        )

                        # Process each tool call
                        for tool_call in tool_calls:
                            tool_name: str = tool_call.get("function", {}).get(
                                "name", ""
                            )
                            tool_id: str = tool_call.get("id", "")

                            # Parse arguments
                            try:
                                tool_args: Dict[str, Any] = json.loads(
                                    tool_call.get("function", {}).get("arguments", "{}")
                                )
                            except (json.JSONDecodeError, TypeError):
                                tool_args = {}

                            logger.info(
                                f"Executing tool: {tool_name} with args: {tool_args}"
                            )

                            # Stream tool call UI
                            html = self._create_tool_call_ui(tool_name, tool_args)
                            yield f"data: {json.dumps({'type': 'html', 'content': html})}\n\n"
                            await asyncio.sleep(0.3)

                            # Show executing indicator
                            executing_html = self._create_executing_ui(tool_name)
                            yield f"data: {json.dumps({'type': 'html', 'content': executing_html})}\n\n"
                            await asyncio.sleep(0.5)

                            # Execute the tool
                            tool_result = await self._execute_tool(
                                tool_name,
                                tool_args,
                                user_id=user_id,
                                session_id=session_id,
                            )
                            logger.info(f"Tool result: {tool_result}")

                            # Show result UI
                            result_html = self._create_result_ui(tool_result)
                            yield f"data: {json.dumps({'type': 'html', 'content': result_html})}\n\n"

                            # Save tool response to database
                            await db.create_chat_message(
                                session_id=session_id,
                                content=tool_result,
                                role=ChatMessageRole.TOOL,
                                tool_call_id=tool_id,
                            )

                            # Add tool result to context
                            context.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_id,
                                    "content": tool_result,
                                }
                            )

                        # Show processing message
                        processing_html = self._create_processing_ui()
                        yield f"data: {json.dumps({'type': 'html', 'content': processing_html})}\n\n"
                        await asyncio.sleep(0.5)

                        # Continue the loop to get final response
                        logger.info("Making follow-up call with tool results...")
                        continue
                    else:
                        # No tool calls, conversation complete
                        logger.info("Conversation complete")
                        break

                except Exception as e:
                    logger.error(f"Error in stream: {str(e)}", exc_info=True)
                    yield f"data: {json.dumps({'type': 'error', 'content': f'Error: {str(e)}'})}\n\n"
                    break

        except Exception as e:
            logger.error(f"Error in stream_chat_response: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': f'Error: {str(e)}'})}\n\n"

    async def _execute_tool(
        self, tool_name: str, parameters: Dict[str, Any], user_id: str, session_id: str
    ) -> str:
        """Execute a tool and return the result.

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            user_id: User ID for authentication
            session_id: Current session ID

        Returns:
            Tool execution result as a string
        """
        # Import tool execution functions
        from backend.server.v2.chat.tools import (
            execute_find_agent,
            execute_get_agent_details,
            execute_setup_agent,
        )

        # Map tool names to execution functions
        tool_functions = {
            "find_agent": execute_find_agent,
            "get_agent_details": execute_get_agent_details,
            "setup_agent": execute_setup_agent,
        }

        # Execute the appropriate tool
        if tool_name in tool_functions:
            tool_func = tool_functions[tool_name]
            return await tool_func(parameters, user_id=user_id, session_id=session_id)
        else:
            return f"Tool '{tool_name}' not implemented"

    def _create_tool_call_ui(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Create HTML UI for tool call display.

        Args:
            tool_name: Name of the tool being called
            tool_args: Arguments passed to the tool

        Returns:
            HTML string for the tool call UI
        """
        return f"""<div class="tool-call-container" style="margin: 20px 0; animation: slideIn 0.3s ease-out;">
            <div class="tool-header" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px; border-radius: 8px 8px 0 0; font-weight: bold; display: flex; align-items: center;">
                <span style="margin-right: 10px;">🔧</span>
                <span>Calling Tool: {tool_name}</span>
            </div>
            <div class="tool-body" style="background: #f7f9fc; padding: 15px; border: 1px solid #e1e4e8; border-top: none; border-radius: 0 0 8px 8px;">
                <div style="color: #586069; font-size: 12px; margin-bottom: 8px;">Parameters:</div>
                <pre style="background: white; padding: 10px; border-radius: 4px; border: 1px solid #e1e4e8; margin: 0; font-size: 13px; color: #24292e;">{json.dumps(tool_args, indent=2)}</pre>
            </div>
        </div>"""

    def _create_executing_ui(self, tool_name: str) -> str:
        """Create HTML UI for tool execution indicator.

        Args:
            tool_name: Name of the tool being executed

        Returns:
            HTML string for the executing UI
        """
        return f"""<div class="tool-executing" style="margin: 10px 0; padding: 10px; background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; color: #856404;">
            <span style="animation: pulse 1.5s infinite;">⏳</span> Executing {tool_name}...
        </div>"""

    def _create_result_ui(self, tool_result: str) -> str:
        """Create HTML UI for tool result display.

        Args:
            tool_result: Result from tool execution

        Returns:
            HTML string for the result UI
        """
        return f"""<div class="tool-result" style="margin: 10px 0; padding: 15px; background: #e8f5e9; border: 1px solid #4caf50; border-radius: 6px;">
            <div style="color: #2e7d32; font-weight: bold; margin-bottom: 8px;">📊 Tool Result:</div>
            <div style="color: #1b5e20;">{tool_result}</div>
        </div>"""

    def _create_processing_ui(self) -> str:
        """Create HTML UI for processing indicator.

        Returns:
            HTML string for the processing UI
        """
        return """<div style="margin: 15px 0; padding: 10px; background: #e3f2fd; border: 1px solid #2196f3; border-radius: 6px; color: #1565c0;">> Processing tool results...</div>"""


# Create a singleton instance
_service_instance: Optional[ChatStreamingService] = None


def get_chat_service(api_key: Optional[str] = None) -> ChatStreamingService:
    """Get or create the chat service instance.

    Args:
        api_key: Optional OpenAI API key

    Returns:
        ChatStreamingService instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = ChatStreamingService(api_key=api_key)
    return _service_instance


async def stream_chat_completion(
    session_id: str,
    user_message: str,
    user_id: str,
    model: str = "gpt-4o",
    max_messages: int = 50,
) -> AsyncGenerator[str, None]:
    """Main entry point for streaming chat completions.

    This function creates a generator that streams OpenAI responses,
    handles tool calling, and streams UI elements back to the route.

    Args:
        session_id: Chat session ID
        user_message: User's input message
        user_id: User ID for authentication
        model: OpenAI model to use
        max_messages: Maximum context messages to include

    Yields:
        SSE formatted JSON strings with response data
    """
    service = get_chat_service()
    async for chunk in service.stream_chat_response(
        session_id=session_id,
        user_message=user_message,
        user_id=user_id,
        model=model,
        max_messages=max_messages,
    ):
        yield chunk
