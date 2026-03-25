"""
Enhanced LLM Block with Context Window Manager Integration

This module provides LLM blocks that intelligently manage conversation context
using the ContextWindowManager for optimal token utilization and message preservation.
"""

import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, SecretStr

from backend.blocks._base import Block, BlockCategory, BlockSchemaInput, BlockSchemaOutput, BlockOutput
from backend.blocks.llm import (
    APIKeyCredentials,
    AICredentials,
    AICredentialsField,
    CredentialsMetaInput,
    LlmModel,
    SchemaField,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    llm_call,
)
from backend.data.execution import ExecutionContext, ConversationContext
from backend.data.model import ModelMetadata
from backend.integrations.providers import ProviderName
from backend.util.context import create_context_manager, MessageImportance

logger = logging.getLogger(__name__)


class ContextAwareLLMInput(BlockSchemaInput):
    """Input schema for context-aware LLM block."""
    prompt: str = SchemaField(description="The prompt to send to the LLM")
    model: LlmModel = SchemaField(
        description="The LLM model to use",
        default=LlmModel.GPT_4_O_MINI
    )
    credentials: AICredentials = AICredentialsField()
    max_tokens: Optional[int] = SchemaField(
        description="Maximum tokens in the response",
        default=1000
    )
    temperature: float = SchemaField(
        description="The temperature parameter for the LLM",
        default=0.7
    )
    context_strategy: str = SchemaField(
        description="Context management strategy",
        default="sliding",
        choices=["sliding", "importance"]
    )
    preserve_system_prompt: bool = SchemaField(
        description="Whether to preserve system prompts in context",
        default=True
    )
    conversation_id: Optional[str] = SchemaField(
        description="Optional conversation ID for context persistence",
        default=None
    )


class ContextAwareLLMOutput(BlockSchemaOutput):
    """Output schema for context-aware LLM block."""
    response: str = SchemaField(description="The LLM's response")
    conversation_id: str = SchemaField(description="The conversation ID")
    context_stats: Dict[str, Any] = SchemaField(
        description="Statistics about the conversation context"
    )
    usage: Dict[str, Any] = SchemaField(description="Token usage information")
    error: str = SchemaField(
        description="Error message if the LLM call fails",
        default=""
    )


class ContextAwareLLMBlock(Block):
    """
    LLM Block with intelligent context window management.
    
    This block automatically manages conversation context to maximize
    useful information while staying within token limits.
    """
    
    def __init__(self):
        super().__init__(
            id="context-aware-llm-001",
            description="LLM block with intelligent context window management",
            categories={BlockCategory.AI},
            input_schema=ContextAwareLLMInput,
            output_schema=ContextAwareLLMOutput,
            test_input={
                "prompt": "What did we discuss about machine learning?",
                "model": LlmModel.GPT_4_O_MINI,
                "credentials": TEST_CREDENTIALS_INPUT,
                "context_strategy": "sliding",
                "preserve_system_prompt": True,
            },
            test_output={
                "response": "Based on our conversation, we discussed various aspects of machine learning including supervised learning algorithms, neural networks, and practical applications.",
                "conversation_id": "conv_12345",
                "context_stats": {
                    "total_messages": 10,
                    "context_messages": 8,
                    "total_tokens": 2500,
                    "max_tokens": 8000,
                    "utilization_percent": 31.25
                },
                "usage": {"prompt_tokens": 500, "completion_tokens": 150, "total_tokens": 650},
                "error": ""
            },
            test_credentials=TEST_CREDENTIALS,
        )
    
    def _get_or_create_conversation_context(
        self,
        execution_context: Optional[ExecutionContext],
        conversation_id: Optional[str],
        max_tokens: int,
        strategy: str
    ) -> ConversationContext:
        """Get or create a conversation context."""
        # Try to get from execution context
        if execution_context and execution_context.conversation_context:
            return execution_context.conversation_context
        
        # Create new context
        context = ConversationContext(
            max_tokens=max_tokens,
            strategy=strategy,
            session_id=conversation_id
        )
        
        # Store in execution context if available
        if execution_context:
            execution_context.conversation_context = context
        
        return context
    
    def _prepare_messages(
        self,
        prompt: str,
        context: ConversationContext,
        preserve_system_prompt: bool
    ) -> List[Dict[str, str]]:
        """Prepare messages for LLM call with context."""
        # Get context messages
        context_messages = context.get_context(format_type="openai")
        
        # Check if we need to add the current prompt
        if context_messages and context_messages[-1]["content"] != prompt:
            # Add current prompt as user message
            context.add_message(
                role="user",
                content=prompt,
                importance="high"  # User prompts are high importance
            )
            context_messages = context.get_context(format_type="openai")
        
        return context_messages
    
    async def run(
        self,
        input_data: ContextAwareLLMInput,
        *,
        credentials: APIKeyCredentials,
        execution_context: Optional[ExecutionContext] = None,
        **kwargs
    ) -> BlockOutput:
        """Execute the context-aware LLM block."""
        try:
            # Get or create conversation context
            max_tokens = input_data.model.context_window or 8000
            context = self._get_or_create_conversation_context(
                execution_context,
                input_data.conversation_id,
                max_tokens,
                input_data.context_strategy
            )
            
            # Prepare messages with context
            messages = self._prepare_messages(
                input_data.prompt,
                context,
                input_data.preserve_system_prompt
            )
            
            # Make LLM call
            response = await llm_call(
                credentials=credentials,
                llm_model=input_data.model,
                prompt=messages,
                max_tokens=input_data.max_tokens,
                temperature=input_data.temperature,
                compress_prompt_to_fit=False,  # Context manager handles this
            )
            
            # Add response to context
            context.add_message(
                role="assistant",
                content=response.response,
                importance="medium"
            )
            
            # Get context statistics
            context_stats = context.get_stats()
            
            # Generate conversation ID if not provided
            conversation_id = input_data.conversation_id or f"conv_{hash(str(context_stats))}"
            
            # Output results
            yield "response", response.response
            yield "conversation_id", conversation_id
            yield "context_stats", context_stats
            yield "usage", {
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
            }
            
            logger.info(
                f"Context-aware LLM call completed. "
                f"Context: {context_stats['context_messages']}/{context_stats['total_messages']} messages, "
                f"Tokens: {context_stats['utilization_percent']:.1f}% utilization"
            )
            
        except Exception as e:
            error_msg = f"Context-aware LLM call failed: {str(e)}"
            logger.error(error_msg)
            yield "error", error_msg
            yield "response", ""
            yield "conversation_id", input_data.conversation_id or ""
            yield "context_stats", {}
            yield "usage", {}


class ContextManagerBlock(Block):
    """
    Block for managing conversation context directly.
    
    This block allows explicit control over conversation context,
    including adding messages, clearing context, and retrieving statistics.
    """
    
    class Input(BlockSchemaInput):
        action: str = SchemaField(
            description="Action to perform",
            choices=["add_message", "get_context", "get_stats", "clear"]
        )
        role: Optional[str] = SchemaField(
            description="Message role (for add_message)",
            default="user"
        )
        content: Optional[str] = SchemaField(
            description="Message content (for add_message)",
            default=""
        )
        importance: Optional[str] = SchemaField(
            description="Message importance (for add_message)",
            default="medium"
        )
        format_type: Optional[str] = SchemaField(
            description="Format type for context (for get_context)",
            default="openai"
        )
        max_tokens: Optional[int] = SchemaField(
            description="Max tokens for context (for get_context)",
            default=8000
        )
        conversation_id: Optional[str] = SchemaField(
            description="Conversation ID",
            default=None
        )
    
    class Output(BlockSchemaOutput):
        success: bool = SchemaField(description="Whether the action succeeded")
        result: Any = SchemaField(description="Result of the action")
        error: str = SchemaField(description="Error message if any", default="")
    
    def __init__(self):
        super().__init__(
            id="context-manager-001",
            description="Block for managing conversation context",
            categories={BlockCategory.AI},
            input_schema=ContextManagerBlock.Input,
            output_schema=ContextManagerBlock.Output,
            test_input={
                "action": "add_message",
                "role": "user",
                "content": "Hello, this is a test message",
                "importance": "high"
            },
            test_output={
                "success": True,
                "result": {"message_added": True},
                "error": ""
            },
        )
    
    async def run(
        self,
        input_data: Input,
        *,
        execution_context: Optional[ExecutionContext] = None,
        **kwargs
    ) -> BlockOutput:
        """Execute the context manager block."""
        try:
            # Get or create context
            context = self._get_or_create_context(execution_context, input_data.conversation_id)
            
            if input_data.action == "add_message":
                if not input_data.content:
                    raise ValueError("Content is required for add_message action")
                
                context.add_message(
                    role=input_data.role or "user",
                    content=input_data.content,
                    importance=input_data.importance or "medium"
                )
                
                yield "success", True
                yield "result", {"message_added": True}
                
            elif input_data.action == "get_context":
                context_messages = context.get_context(
                    format_type=input_data.format_type or "openai",
                    max_tokens=input_data.max_tokens
                )
                
                yield "success", True
                yield "result", {"messages": context_messages}
                
            elif input_data.action == "get_stats":
                stats = context.get_stats()
                yield "success", True
                yield "result", stats
                
            elif input_data.action == "clear":
                context.clear()
                yield "success", True
                yield "result", {"context_cleared": True}
                
            else:
                raise ValueError(f"Unknown action: {input_data.action}")
                
        except Exception as e:
            error_msg = f"Context manager action failed: {str(e)}"
            logger.error(error_msg)
            yield "success", False
            yield "result", None
            yield "error", error_msg
    
    def _get_or_create_context(
        self,
        execution_context: Optional[ExecutionContext],
        conversation_id: Optional[str]
    ) -> ConversationContext:
        """Get or create conversation context."""
        if execution_context and execution_context.conversation_context:
            return execution_context.conversation_context
        
        context = ConversationContext(
            max_tokens=8000,
            strategy="sliding",
            session_id=conversation_id
        )
        
        if execution_context:
            execution_context.conversation_context = context
        
        return context


# Export the blocks
__all__ = [
    "ContextAwareLLMBlock",
    "ContextManagerBlock",
]
