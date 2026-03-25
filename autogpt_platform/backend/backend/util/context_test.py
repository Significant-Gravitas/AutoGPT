"""
Tests for Context Window Manager
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import List

from backend.util.context import (
    ContextMessage,
    ContextWindowManager,
    MessageImportance,
    MessageRole,
    SlidingWindowStrategy,
    ImportanceBasedStrategy,
    get_context_manager,
    create_context_manager,
)


class TestContextMessage:
    """Test ContextMessage class."""
    
    def test_message_creation(self):
        """Test creating a context message."""
        msg = ContextMessage(
            role=MessageRole.USER,
            content="Hello, world!",
            importance=MessageImportance.HIGH
        )
        
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, world!"
        assert msg.importance == MessageImportance.HIGH
        assert msg.message_id.startswith("user_")
        assert isinstance(msg.timestamp, datetime)
    
    def test_message_id_generation(self):
        """Test unique ID generation."""
        msg1 = ContextMessage(
            role=MessageRole.USER,
            content="Test message"
        )
        msg2 = ContextMessage(
            role=MessageRole.USER,
            content="Test message"
        )
        
        # Different timestamps should generate different IDs
        assert msg1.message_id != msg2.message_id
    
    def test_token_count_estimation(self):
        """Test token count estimation."""
        msg = ContextMessage(
            role=MessageRole.USER,
            content="This is a test message with multiple words."
        )
        
        # Should estimate tokens (roughly 1 token per 4 characters)
        assert msg.token_count > 0
        assert msg.token_count == len(msg.content) // 4 or msg.token_count == len(msg.content) // 4 + 1


class TestSlidingWindowStrategy:
    """Test SlidingWindowStrategy class."""
    
    def test_basic_selection(self):
        """Test basic message selection."""
        strategy = SlidingWindowStrategy(preserve_recent=3)
        
        messages = [
            ContextMessage(role=MessageRole.USER, content=f"Message {i}", token_count=10)
            for i in range(10)
        ]
        
        selected, stats = strategy.select_messages(messages, max_tokens=50)
        
        # Should preserve recent messages
        assert len(selected) <= 5  # 50 tokens / 10 per message
        assert selected[-1].content == "Message 9"  # Last message preserved
    
    def test_critical_message_preservation(self):
        """Test that critical messages are always preserved."""
        strategy = SlidingWindowStrategy()
        
        messages = [
            ContextMessage(
                role=MessageRole.SYSTEM,
                content="Critical system prompt",
                importance=MessageImportance.CRITICAL,
                token_count=20
            ),
            ContextMessage(
                role=MessageRole.USER,
                content="Regular message",
                importance=MessageImportance.MEDIUM,
                token_count=10
            )
        ]
        
        selected, stats = strategy.select_messages(messages, max_tokens=25)
        
        # Critical message should be preserved even if it uses most tokens
        assert len(selected) == 2
        assert selected[0].importance == MessageImportance.CRITICAL
    
    def test_importance_preservation(self):
        """Test that importance affects preservation."""
        strategy = SlidingWindowStrategy(preserve_importance=True)
        
        messages = [
            ContextMessage(
                role=MessageRole.USER,
                content="Low importance",
                importance=MessageImportance.LOW,
                token_count=10
            ),
            ContextMessage(
                role=MessageRole.USER,
                content="High importance",
                importance=MessageImportance.HIGH,
                token_count=10
            )
        ]
        
        selected, stats = strategy.select_messages(messages, max_tokens=10)
        
        # Should prefer high importance message
        assert len(selected) == 1
        assert selected[0].importance == MessageImportance.HIGH


class TestImportanceBasedStrategy:
    """Test ImportanceBasedStrategy class."""
    
    def test_score_calculation(self):
        """Test importance score calculation."""
        strategy = ImportanceBasedStrategy()
        
        now = datetime.now(timezone.utc)
        messages = [
            ContextMessage(
                role=MessageRole.SYSTEM,
                content="Critical message",
                importance=MessageImportance.CRITICAL,
                timestamp=now,
                token_count=10
            ),
            ContextMessage(
                role=MessageRole.USER,
                content="Old low importance",
                importance=MessageImportance.LOW,
                timestamp=now - timedelta(hours=10),
                token_count=10
            )
        ]
        
        selected, stats = strategy.select_messages(messages, max_tokens=20)
        
        # Should select both, but critical first
        assert len(selected) == 2
        assert selected[0].importance == MessageImportance.CRITICAL
    
    def test_time_decay(self):
        """Test time decay factor."""
        strategy = ImportanceBasedStrategy(time_decay_factor=1.0)  # High decay
        
        now = datetime.now(timezone.utc)
        messages = [
            ContextMessage(
                role=MessageRole.USER,
                content="Recent message",
                importance=MessageImportance.MEDIUM,
                timestamp=now,
                token_count=10
            ),
            ContextMessage(
                role=MessageRole.USER,
                content="Old message",
                importance=MessageImportance.HIGH,
                timestamp=now - timedelta(hours=50),  # Very old
                token_count=10
            )
        ]
        
        selected, stats = strategy.select_messages(messages, max_tokens=10)
        
        # Should prefer recent message despite lower importance
        assert len(selected) == 1
        assert selected[0].content == "Recent message"


class TestContextWindowManager:
    """Test ContextWindowManager class."""
    
    def test_initialization(self):
        """Test manager initialization."""
        manager = ContextWindowManager(max_tokens=1000, strategy="sliding")
        
        assert manager.max_tokens == 1000
        assert isinstance(manager.strategy, SlidingWindowStrategy)
        assert len(manager.messages) == 0
    
    def test_add_message(self):
        """Test adding messages."""
        manager = ContextWindowManager()
        
        manager.add_message(MessageRole.USER, "Hello world")
        manager.add_message(MessageRole.ASSISTANT, "Hi there!")
        
        assert len(manager.messages) == 2
        assert manager.messages[0].role == MessageRole.USER
        assert manager.messages[1].role == MessageRole.ASSISTANT
    
    def test_auto_importance_detection(self):
        """Test automatic importance detection."""
        manager = ContextWindowManager()
        
        # System message with main objective should be critical
        manager.add_message(
            MessageRole.SYSTEM,
            f"{MAIN_OBJECTIVE_PREFIX} You are a helpful assistant."
        )
        
        assert manager.messages[0].importance == MessageImportance.CRITICAL
        
        # Message with important keyword should be high
        manager.add_message(
            MessageRole.USER,
            "Remember to always be polite."
        )
        
        assert manager.messages[1].importance == MessageImportance.HIGH
    
    def test_get_context(self):
        """Test getting context within limits."""
        manager = ContextWindowManager(max_tokens=50, strategy="sliding")
        
        # Add messages that exceed the limit
        for i in range(10):
            manager.add_message(MessageRole.USER, f"Message {i}", token_count=10)
        
        context, stats = manager.get_context()
        
        # Should be within token limit
        assert sum(m.token_count for m in context) <= 50
        assert stats.total_messages == 10
        assert stats.max_tokens == 50
        assert stats.preserved_messages > 0
    
    def test_context_caching(self):
        """Test context caching."""
        manager = ContextWindowManager()
        
        manager.add_message(MessageRole.USER, "Test message")
        
        # First call should compute
        context1, stats1 = manager.get_context()
        
        # Second call should use cache
        context2, stats2 = manager.get_context()
        
        assert context1 is context2  # Same object from cache
        assert stats1 is stats2
        
        # Force refresh should recompute
        context3, stats3 = manager.get_context(force_refresh=True)
        assert context3 is not context1
    
    def test_formatted_context(self):
        """Test getting formatted context."""
        manager = ContextWindowManager()
        
        manager.add_message(MessageRole.SYSTEM, "You are helpful.")
        manager.add_message(MessageRole.USER, "Hello!")
        manager.add_message(MessageRole.ASSISTANT, "Hi there!")
        
        # OpenAI format
        openai_context = manager.get_formatted_context("openai")
        assert openai_context[0]["role"] == "system"
        assert openai_context[1]["role"] == "user"
        assert openai_context[2]["role"] == "assistant"
        
        # Anthropic format
        anthropic_context = manager.get_formatted_context("anthropic")
        assert len(anthropic_context) == 2  # System + combined others
        assert anthropic_context[0]["role"] == "system"
    
    def test_clear_context(self):
        """Test clearing context."""
        manager = ContextWindowManager()
        
        manager.add_message(MessageRole.USER, "Test message")
        assert len(manager.messages) == 1
        
        manager.clear()
        assert len(manager.messages) == 0
        assert manager._cached_context is None
    
    def test_token_count(self):
        """Test token count tracking."""
        manager = ContextWindowManager()
        
        manager.add_message(MessageRole.USER, "Hello", token_count=2)
        manager.add_message(MessageRole.ASSISTANT, "Hi there!", token_count=3)
        
        assert manager.get_token_count() == 5
        assert manager.get_message_count() == 2


class TestContextManagerFactory:
    """Test context manager factory functions."""
    
    def test_get_context_manager(self):
        """Test getting default context manager."""
        manager1 = get_context_manager()
        manager2 = get_context_manager()
        
        # Should return same instance
        assert manager1 is manager2
    
    def test_create_context_manager(self):
        """Test creating new context manager."""
        manager1 = create_context_manager(max_tokens=1000)
        manager2 = create_context_manager(max_tokens=2000)
        
        # Should return different instances
        assert manager1 is not manager2
        assert manager1.max_tokens == 1000
        assert manager2.max_tokens == 2000


# Fixtures for testing
@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        ContextMessage(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant.",
            importance=MessageImportance.CRITICAL,
            token_count=10
        ),
        ContextMessage(
            role=MessageRole.USER,
            content="What is the capital of France?",
            importance=MessageImportance.MEDIUM,
            token_count=8
        ),
        ContextMessage(
            role=MessageRole.ASSISTANT,
            content="The capital of France is Paris.",
            importance=MessageImportance.MEDIUM,
            token_count=9
        ),
        ContextMessage(
            role=MessageRole.USER,
            content="Tell me more about Paris.",
            importance=MessageImportance.HIGH,
            token_count=7
        ),
        ContextMessage(
            role=MessageRole.ASSISTANT,
            content="Paris is known for the Eiffel Tower...",
            importance=MessageImportance.MEDIUM,
            token_count=12
        ),
    ]


@pytest.fixture
def context_manager():
    """Create a context manager for testing."""
    return ContextWindowManager(max_tokens=50, strategy="sliding")


class TestIntegration:
    """Integration tests for context window manager."""
    
    def test_full_conversation_flow(self, context_manager, sample_messages):
        """Test a full conversation flow."""
        # Add all messages
        for msg in sample_messages:
            context_manager.add_message(
                msg.role,
                msg.content,
                msg.importance,
                token_count=msg.token_count
            )
        
        # Get context
        context, stats = context_manager.get_context()
        
        # Verify context is within limits
        total_tokens = sum(m.token_count for m in context)
        assert total_tokens <= 50
        
        # Critical message should be preserved
        critical_messages = [m for m in context if m.importance == MessageImportance.CRITICAL]
        assert len(critical_messages) > 0
        
        # Get formatted for different providers
        openai_format = context_manager.get_formatted_context("openai")
        anthropic_format = context_manager.get_formatted_context("anthropic")
        
        assert len(openai_format) == len(context)
        assert len(anthropic_format) <= len(context)  # System messages combined
    
    def test_long_conversation_truncation(self):
        """Test truncation of long conversations."""
        manager = ContextWindowManager(max_tokens=100, strategy="sliding")
        
        # Add many messages
        for i in range(50):
            manager.add_message(
                MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
                f"This is message number {i}. " + "x" * 20,  # Make it longer
                token_count=15
            )
        
        context, stats = manager.get_context()
        
        # Should truncate to fit within limit
        assert sum(m.token_count for m in context) <= 100
        assert stats.total_messages == 50
        assert stats.preserved_messages < 50
        
        # Should preserve recent messages
        assert context[-1].content == "This is message number 49. " + "x" * 20
    
    def test_importance_strategy_performance(self):
        """Test importance-based strategy performance."""
        manager = ContextWindowManager(max_tokens=100, strategy="importance")
        
        # Add messages with varying importance
        messages_data = [
            (MessageRole.SYSTEM, "Critical system prompt", MessageImportance.CRITICAL),
            (MessageRole.USER, "Important question", MessageImportance.HIGH),
            (MessageRole.USER, "Regular chat", MessageImportance.MEDIUM),
            (MessageRole.USER, "Low value comment", MessageImportance.LOW),
            (MessageRole.ASSISTANT, "Helpful response", MessageImportance.MEDIUM),
        ]
        
        for role, content, importance in messages_data:
            manager.add_message(role, content, importance, token_count=25)
        
        context, stats = manager.get_context()
        
        # Should prioritize important messages
        importance_order = [m.importance for m in context]
        assert MessageImportance.CRITICAL in importance_order
        
        # Critical and high importance should be preserved
        critical_count = sum(1 for m in context if m.importance == MessageImportance.CRITICAL)
        high_count = sum(1 for m in context if m.importance == MessageImportance.HIGH)
        assert critical_count > 0
        assert high_count > 0
