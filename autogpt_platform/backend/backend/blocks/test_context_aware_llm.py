"""
Tests for Context-Aware LLM Blocks
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from backend.blocks.context_aware_llm import (
    ContextAwareLLMBlock,
    ContextManagerBlock,
    ConversationContext,
)
from backend.blocks.llm import LlmModel, APIKeyCredentials
from backend.data.execution import ExecutionContext
from backend.data.model import CredentialsMetaInput


class TestConversationContext:
    """Test ConversationContext integration."""
    
    def test_context_creation(self):
        """Test creating a conversation context."""
        context = ConversationContext(
            max_tokens=4000,
            strategy="sliding",
            session_id="test_session"
        )
        
        assert context.session_id == "test_session"
        assert context.message_count == 0
        assert context.context_manager is not None
    
    def test_add_message(self):
        """Test adding messages to context."""
        context = ConversationContext()
        
        context.add_message("user", "Hello world", "high")
        assert context.message_count == 1
        
        context.add_message("assistant", "Hi there!", "medium")
        assert context.message_count == 2
    
    def test_get_context(self):
        """Test getting formatted context."""
        context = ConversationContext()
        
        context.add_message("system", "You are helpful.")
        context.add_message("user", "Hello!")
        
        openai_context = context.get_context("openai")
        assert len(openai_context) == 2
        assert openai_context[0]["role"] == "system"
        assert openai_context[1]["role"] == "user"
        
        anthropic_context = context.get_context("anthropic")
        assert len(anthropic_context) <= 2  # System messages might be combined
    
    def test_context_stats(self):
        """Test getting context statistics."""
        context = ConversationContext(max_tokens=1000)
        
        # Add some messages
        for i in range(5):
            context.add_message("user", f"Message {i}")
        
        stats = context.get_stats()
        assert "total_messages" in stats
        assert "context_messages" in stats
        assert "total_tokens" in stats
        assert "max_tokens" in stats
        assert "utilization_percent" in stats
        assert stats["total_messages"] == 5
        assert stats["max_tokens"] == 1000
    
    def test_clear_context(self):
        """Test clearing context."""
        context = ConversationContext()
        
        context.add_message("user", "Test message")
        assert context.message_count == 1
        
        context.clear()
        assert context.message_count == 0


class TestContextAwareLLMBlock:
    """Test ContextAwareLLMBlock."""
    
    @pytest.fixture
    def block(self):
        """Create a ContextAwareLLMBlock instance."""
        return ContextAwareLLMBlock()
    
    @pytest.fixture
    def mock_credentials(self):
        """Create mock credentials."""
        return APIKeyCredentials(
            id="test",
            provider="openai",
            api_key="sk-test-key",
            title="Test",
            metadata=CredentialsMetaInput()
        )
    
    @pytest.fixture
    def execution_context(self):
        """Create a mock execution context."""
        return ExecutionContext(
            user_id="test_user",
            graph_id="test_graph",
            graph_exec_id="test_exec"
        )
    
    @pytest.mark.asyncio
    async def test_block_initialization(self, block):
        """Test block initialization."""
        assert block.id == "context-aware-llm-001"
        assert "context window management" in block.description.lower()
        assert block.categories == {BlockCategory.AI}
    
    @pytest.mark.asyncio
    async def test_get_or_create_conversation_context(
        self,
        block,
        execution_context
    ):
        """Test getting or creating conversation context."""
        # First call should create new context
        context1 = block._get_or_create_conversation_context(
            execution_context,
            "conv_123",
            4000,
            "sliding"
        )
        assert isinstance(context1, ConversationContext)
        assert context1.session_id == "conv_123"
        
        # Second call should return same context
        context2 = block._get_or_create_conversation_context(
            execution_context,
            "conv_123",
            4000,
            "sliding"
        )
        assert context1 is context2
        assert execution_context.conversation_context is context1
    
    @pytest.mark.asyncio
    async def test_prepare_messages(self, block):
        """Test preparing messages with context."""
        context = ConversationContext()
        
        # Add some initial context
        context.add_message("system", "You are helpful.")
        context.add_message("user", "Previous question")
        context.add_message("assistant", "Previous answer")
        
        # Prepare new messages
        messages = block._prepare_messages(
            "New question",
            context,
            preserve_system_prompt=True
        )
        
        # Should include context plus new prompt
        assert len(messages) >= 4
        assert messages[-1]["content"] == "New question"
        assert messages[-1]["role"] == "user"
    
    @pytest.mark.asyncio
    @patch('backend.blocks.context_aware_llm.llm_call')
    async def test_run_success(
        self,
        mock_llm_call,
        block,
        mock_credentials,
        execution_context
    ):
        """Test successful block execution."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.response = "Test response"
        mock_response.prompt_tokens = 10
        mock_response.completion_tokens = 5
        mock_response.total_tokens = 15
        mock_llm_call.return_value = mock_response
        
        # Create input
        input_data = block.input_schema(
            prompt="Test prompt",
            model=LlmModel.GPT_4_O_MINI,
            credentials=mock_credentials,
            context_strategy="sliding"
        )
        
        # Run block
        results = []
        async for result in block.run(
            input_data,
            credentials=mock_credentials,
            execution_context=execution_context
        ):
            results.append(result)
        
        # Check results
        assert "response" in results
        assert "conversation_id" in results
        assert "context_stats" in results
        assert "usage" in results
        assert results[0][1] == "Test response"
        
        # Check context was updated
        assert execution_context.conversation_context is not None
        assert execution_context.conversation_context.message_count >= 2  # prompt + response
    
    @pytest.mark.asyncio
    @patch('backend.blocks.context_aware_llm.llm_call')
    async def test_run_with_existing_context(
        self,
        mock_llm_call,
        block,
        mock_credentials,
        execution_context
    ):
        """Test running with existing conversation context."""
        # Create existing context
        existing_context = ConversationContext(
            max_tokens=4000,
            strategy="sliding",
            session_id="existing_conv"
        )
        existing_context.add_message("user", "Previous message")
        existing_context.add_message("assistant", "Previous response")
        execution_context.conversation_context = existing_context
        
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.response = "New response"
        mock_response.prompt_tokens = 20
        mock_response.completion_tokens = 10
        mock_response.total_tokens = 30
        mock_llm_call.return_value = mock_response
        
        # Create input
        input_data = block.input_schema(
            prompt="Follow-up question",
            model=LlmModel.GPT_4_O_MINI,
            credentials=mock_credentials,
            conversation_id="existing_conv"
        )
        
        # Run block
        results = []
        async for result in block.run(
            input_data,
            credentials=mock_credentials,
            execution_context=execution_context
        ):
            results.append(result)
        
        # Check that existing context was used
        assert execution_context.conversation_context is existing_context
        assert execution_context.conversation_context.message_count == 4  # 2 existing + prompt + response
    
    @pytest.mark.asyncio
    async def test_run_error_handling(
        self,
        block,
        mock_credentials,
        execution_context
    ):
        """Test error handling in block execution."""
        # Create invalid input
        input_data = block.input_schema(
            prompt="Test prompt",
            model=LlmModel.GPT_4_O_MINI,
            credentials=mock_credentials,
            context_strategy="invalid_strategy"  # This might cause issues
        )
        
        # Run block
        results = []
        async for result in block.run(
            input_data,
            credentials=mock_credentials,
            execution_context=execution_context
        ):
            results.append(result)
        
        # Should handle errors gracefully
        assert "error" in results
        assert "response" in results


class TestContextManagerBlock:
    """Test ContextManagerBlock."""
    
    @pytest.fixture
    def block(self):
        """Create a ContextManagerBlock instance."""
        return ContextManagerBlock()
    
    @pytest.fixture
    def execution_context(self):
        """Create a mock execution context."""
        return ExecutionContext(
            user_id="test_user",
            graph_id="test_graph"
        )
    
    @pytest.mark.asyncio
    async def test_block_initialization(self, block):
        """Test block initialization."""
        assert block.id == "context-manager-001"
        assert "managing conversation context" in block.description.lower()
        assert block.categories == {BlockCategory.AI}
    
    @pytest.mark.asyncio
    async def test_add_message_action(
        self,
        block,
        execution_context
    ):
        """Test adding a message action."""
        input_data = block.input_schema(
            action="add_message",
            role="user",
            content="Test message",
            importance="high"
        )
        
        results = []
        async for result in block.run(
            input_data,
            execution_context=execution_context
        ):
            results.append(result)
        
        assert results[0][1] is True  # success
        assert results[1][1]["message_added"] is True  # result
        assert execution_context.conversation_context is not None
        assert execution_context.conversation_context.message_count == 1
    
    @pytest.mark.asyncio
    async def test_get_context_action(
        self,
        block,
        execution_context
    ):
        """Test getting context action."""
        # First add some messages
        context = ConversationContext()
        context.add_message("user", "Hello")
        context.add_message("assistant", "Hi!")
        execution_context.conversation_context = context
        
        input_data = block.input_schema(
            action="get_context",
            format_type="openai"
        )
        
        results = []
        async for result in block.run(
            input_data,
            execution_context=execution_context
        ):
            results.append(result)
        
        assert results[0][1] is True  # success
        assert "messages" in results[1][1]  # result
        assert len(results[1][1]["messages"]) == 2
    
    @pytest.mark.asyncio
    async def test_get_stats_action(
        self,
        block,
        execution_context
    ):
        """Test getting stats action."""
        # Create context with messages
        context = ConversationContext(max_tokens=1000)
        for i in range(5):
            context.add_message("user", f"Message {i}")
        execution_context.conversation_context = context
        
        input_data = block.input_schema(action="get_stats")
        
        results = []
        async for result in block.run(
            input_data,
            execution_context=execution_context
        ):
            results.append(result)
        
        assert results[0][1] is True  # success
        stats = results[1][1]  # result
        assert "total_messages" in stats
        assert stats["total_messages"] == 5
    
    @pytest.mark.asyncio
    async def test_clear_action(
        self,
        block,
        execution_context
    ):
        """Test clear action."""
        # Create context with messages
        context = ConversationContext()
        context.add_message("user", "Test")
        execution_context.conversation_context = context
        
        input_data = block.input_schema(action="clear")
        
        results = []
        async for result in block.run(
            input_data,
            execution_context=execution_context
        ):
            results.append(result)
        
        assert results[0][1] is True  # success
        assert results[1][1]["context_cleared"] is True  # result
        assert context.message_count == 0
    
    @pytest.mark.asyncio
    async def test_invalid_action(
        self,
        block,
        execution_context
    ):
        """Test handling invalid action."""
        input_data = block.input_schema(action="invalid_action")
        
        results = []
        async for result in block.run(
            input_data,
            execution_context=execution_context
        ):
            results.append(result)
        
        assert results[0][1] is False  # success
        assert "error" in results[2][1]  # error message
        assert "Unknown action" in results[2][1]
    
    @pytest.mark.asyncio
    async def test_add_message_without_content(
        self,
        block,
        execution_context
    ):
        """Test adding message without content."""
        input_data = block.input_schema(
            action="add_message",
            role="user"
            # Missing content
        )
        
        results = []
        async for result in block.run(
            input_data,
            execution_context=execution_context
        ):
            results.append(result)
        
        assert results[0][1] is False  # success
        assert "Content is required" in results[2][1]  # error


class TestIntegration:
    """Integration tests for context-aware blocks."""
    
    @pytest.mark.asyncio
    async def test_conversation_flow(self):
        """Test a complete conversation flow."""
        # Create execution context
        exec_context = ExecutionContext(user_id="test_user")
        
        # Create context manager block
        context_block = ContextManagerBlock()
        
        # Add initial system message
        input_data = context_block.input_schema(
            action="add_message",
            role="system",
            content="You are a helpful assistant.",
            importance="critical"
        )
        
        async for result in context_block.run(input_data, execution_context=exec_context):
            pass  # System message added
        
        # Verify context
        assert exec_context.conversation_context is not None
        assert exec_context.conversation_context.message_count == 1
        
        # Get context stats
        input_data = context_block.input_schema(action="get_stats")
        results = []
        async for result in context_block.run(input_data, execution_context=exec_context):
            results.append(result)
        
        stats = results[1][1]
        assert stats["total_messages"] == 1
        
        # Clear context
        input_data = context_block.input_schema(action="clear")
        async for result in context_block.run(input_data, execution_context=exec_context):
            pass
        
        assert exec_context.conversation_context.message_count == 0
