"""
Context Window Manager for AutoGPT

Provides intelligent context management for LLM conversations including:
- Sliding window context management
- Important message preservation
- Token-aware truncation
- Conversation summarization
- Context optimization strategies
"""

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, Union

from pydantic import BaseModel

from backend.util.prompt import MAIN_OBJECTIVE_PREFIX
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()


class MessageRole(str, Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


class MessageImportance(str, Enum):
    """Message importance levels for preservation."""
    CRITICAL = "critical"  # System prompts, main objectives
    HIGH = "high"  # Important instructions, key decisions
    MEDIUM = "medium"  # Regular conversation, context
    LOW = "low"  # Redundant, verbose, or less important


@dataclass
class ContextMessage:
    """A message in the conversation context."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    importance: MessageImportance = MessageImportance.MEDIUM
    token_count: int = 0
    message_id: str = field(default="")
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.message_id:
            # Generate unique ID based on content and timestamp
            content_hash = hashlib.md5(
                f"{self.role}:{self.content}:{self.timestamp.isoformat()}".encode()
            ).hexdigest()[:8]
            self.message_id = f"{self.role.value}_{content_hash}"


@dataclass
class ContextWindowStats:
    """Statistics about the context window."""
    total_messages: int = 0
    total_tokens: int = 0
    max_tokens: int = 0
    utilization_percent: float = 0.0
    preserved_messages: int = 0
    summarized_messages: int = 0
    truncated_messages: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ContextStrategy(Protocol):
    """Protocol for context management strategies."""
    
    def select_messages(
        self,
        messages: List[ContextMessage],
        max_tokens: int,
        **kwargs
    ) -> Tuple[List[ContextMessage], ContextWindowStats]:
        """Select messages to fit within token limit."""
        ...


class SlidingWindowStrategy:
    """Sliding window context management strategy."""
    
    def __init__(
        self,
        reserve_tokens: int = 1000,  # Reserve for response
        preserve_recent: int = 10,  # Always keep last N messages
        preserve_importance: bool = True,
    ):
        self.reserve_tokens = reserve_tokens
        self.preserve_recent = preserve_recent
        self.preserve_importance = preserve_importance
    
    def select_messages(
        self,
        messages: List[ContextMessage],
        max_tokens: int,
        **kwargs
    ) -> Tuple[List[ContextMessage], ContextWindowStats]:
        """Select messages using sliding window approach."""
        available_tokens = max_tokens - self.reserve_tokens
        selected_messages = []
        used_tokens = 0
        
        # Always preserve critical messages
        critical_messages = [m for m in messages if m.importance == MessageImportance.CRITICAL]
        for msg in critical_messages:
            if used_tokens + msg.token_count <= available_tokens:
                selected_messages.append(msg)
                used_tokens += msg.token_count
        
        # Preserve recent messages
        recent_messages = messages[-self.preserve_recent:] if len(messages) > self.preserve_recent else messages
        for msg in recent_messages:
            if msg not in selected_messages and used_tokens + msg.token_count <= available_tokens:
                selected_messages.append(msg)
                used_tokens += msg.token_count
        
        # Fill remaining space with other messages by importance
        other_messages = [
            m for m in messages 
            if m not in selected_messages 
            and m.importance != MessageImportance.CRITICAL
        ]
        
        # Sort by importance and recency
        other_messages.sort(
            key=lambda m: (
                0 if m.importance == MessageImportance.HIGH else 1,
                0 if m.importance == MessageImportance.MEDIUM else 2,
                -m.timestamp.timestamp()
            )
        )
        
        for msg in other_messages:
            if used_tokens + msg.token_count <= available_tokens:
                selected_messages.append(msg)
                used_tokens += msg.token_count
        
        # Sort selected messages by timestamp
        selected_messages.sort(key=lambda m: m.timestamp)
        
        # Calculate stats
        stats = ContextWindowStats(
            total_messages=len(messages),
            total_tokens=sum(m.token_count for m in messages),
            max_tokens=max_tokens,
            utilization_percent=(used_tokens / max_tokens) * 100,
            preserved_messages=len(selected_messages),
        )
        
        return selected_messages, stats


class ImportanceBasedStrategy:
    """Importance-based context management strategy."""
    
    def __init__(
        self,
        importance_weights: Dict[MessageImportance, float] = None,
        time_decay_factor: float = 0.1,
    ):
        self.importance_weights = importance_weights or {
            MessageImportance.CRITICAL: 1.0,
            MessageImportance.HIGH: 0.8,
            MessageImportance.MEDIUM: 0.5,
            MessageImportance.LOW: 0.2,
        }
        self.time_decay_factor = time_decay_factor
    
    def select_messages(
        self,
        messages: List[ContextMessage],
        max_tokens: int,
        **kwargs
    ) -> Tuple[List[ContextMessage], ContextWindowStats]:
        """Select messages based on importance scores."""
        available_tokens = max_tokens - 1000  # Reserve for response
        
        # Calculate scores for each message
        scored_messages = []
        current_time = time.time()
        
        for msg in messages:
            # Base score from importance
            base_score = self.importance_weights.get(msg.importance, 0.5)
            
            # Apply time decay (older messages lose score)
            age_hours = (current_time - msg.timestamp.timestamp()) / 3600
            time_factor = max(0.1, 1.0 - (age_hours * self.time_decay_factor / 100))
            
            final_score = base_score * time_factor
            scored_messages.append((msg, final_score))
        
        # Sort by score (descending)
        scored_messages.sort(key=lambda x: x[1], reverse=True)
        
        # Select messages within token limit
        selected_messages = []
        used_tokens = 0
        
        for msg, score in scored_messages:
            if used_tokens + msg.token_count <= available_tokens:
                selected_messages.append(msg)
                used_tokens += msg.token_count
        
        # Sort by timestamp
        selected_messages.sort(key=lambda m: m.timestamp)
        
        # Calculate stats
        stats = ContextWindowStats(
            total_messages=len(messages),
            total_tokens=sum(m.token_count for m in messages),
            max_tokens=max_tokens,
            utilization_percent=(used_tokens / max_tokens) * 100,
            preserved_messages=len(selected_messages),
        )
        
        return selected_messages, stats


class ContextWindowManager:
    """Manages conversation context with intelligent truncation and preservation."""
    
    def __init__(
        self,
        max_tokens: int = 8000,
        strategy: Union[str, ContextStrategy] = "sliding",
        token_counter: Optional[callable] = None,
        enable_summarization: bool = False,
    ):
        """
        Initialize the context window manager.
        
        Args:
            max_tokens: Maximum tokens allowed in context
            strategy: Context management strategy ("sliding", "importance", or custom)
            token_counter: Function to count tokens (defaults to simple estimation)
            enable_summarization: Whether to enable message summarization
        """
        self.max_tokens = max_tokens
        self.enable_summarization = enable_summarization
        self.token_counter = token_counter or self._default_token_counter
        
        # Initialize strategy
        if isinstance(strategy, str):
            if strategy == "sliding":
                self.strategy = SlidingWindowStrategy()
            elif strategy == "importance":
                self.strategy = ImportanceBasedStrategy()
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        else:
            self.strategy = strategy
        
        # Message storage
        self.messages: List[ContextMessage] = []
        self.stats = ContextWindowStats(max_tokens=max_tokens)
        
        # Caching
        self._cached_context: Optional[List[ContextMessage]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds = 30
    
    def add_message(
        self,
        role: Union[MessageRole, str],
        content: str,
        importance: Union[MessageImportance, str] = MessageImportance.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a message to the conversation context."""
        # Convert string enums
        if isinstance(role, str):
            role = MessageRole(role)
        if isinstance(importance, str):
            importance = MessageImportance(importance)
        
        # Auto-detect importance for certain patterns
        if importance == MessageImportance.MEDIUM:
            importance = self._detect_importance(role, content)
        
        # Create message
        message = ContextMessage(
            role=role,
            content=content,
            importance=importance,
            token_count=self.token_counter(content),
            metadata=metadata or {}
        )
        
        # Add to messages
        self.messages.append(message)
        
        # Invalidate cache
        self._cached_context = None
        self._cache_timestamp = None
        
        logger.debug(f"Added {role.value} message with {message.token_count} tokens")
    
    def get_context(
        self,
        max_tokens: Optional[int] = None,
        force_refresh: bool = False
    ) -> Tuple[List[ContextMessage], ContextWindowStats]:
        """
        Get the current context within token limits.
        
        Args:
            max_tokens: Override default max tokens
            force_refresh: Force refresh of cached context
            
        Returns:
            Tuple of (selected_messages, stats)
        """
        # Check cache
        if (
            not force_refresh
            and self._cached_context is not None
            and self._cache_timestamp is not None
            and (datetime.now(timezone.utc) - self._cache_timestamp).seconds < self._cache_ttl_seconds
        ):
            return self._cached_context, self.stats
        
        # Use provided max_tokens or default
        target_tokens = max_tokens or self.max_tokens
        
        # Apply strategy to select messages
        selected_messages, new_stats = self.strategy.select_messages(
            self.messages,
            target_tokens
        )
        
        # Update stats
        new_stats.max_tokens = target_tokens
        new_stats.last_updated = datetime.now(timezone.utc)
        self.stats = new_stats
        
        # Cache result
        self._cached_context = selected_messages
        self._cache_timestamp = datetime.now(timezone.utc)
        
        return selected_messages, self.stats
    
    def get_formatted_context(
        self,
        format_type: str = "openai",
        max_tokens: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Get context formatted for specific LLM providers.
        
        Args:
            format_type: Format type ("openai", "anthropic", "llama")
            max_tokens: Override max tokens
            
        Returns:
            Formatted messages list
        """
        messages, _ = self.get_context(max_tokens)
        
        if format_type == "openai":
            return [
                {"role": msg.role.value, "content": msg.content}
                for msg in messages
            ]
        elif format_type == "anthropic":
            # Anthropic format
            formatted = []
            system_messages = [msg for msg in messages if msg.role == MessageRole.SYSTEM]
            other_messages = [msg for msg in messages if msg.role != MessageRole.SYSTEM]
            
            if system_messages:
                # Combine system messages
                system_content = "\n\n".join(msg.content for msg in system_messages)
                formatted.append({"role": "system", "content": system_content})
            
            # Add other messages
            for msg in other_messages:
                formatted.append({"role": msg.role.value, "content": msg.content})
            
            return formatted
        else:
            # Default to OpenAI format
            return [
                {"role": msg.role.value, "content": msg.content}
                for msg in messages
            ]
    
    def clear(self) -> None:
        """Clear all messages from context."""
        self.messages.clear()
        self._cached_context = None
        self._cache_timestamp = None
        logger.info("Context window cleared")
    
    def get_message_count(self) -> int:
        """Get total number of messages in context."""
        return len(self.messages)
    
    def get_token_count(self) -> int:
        """Get total token count of all messages."""
        return sum(m.token_count for m in self.messages)
    
    def _detect_importance(
        self,
        role: MessageRole,
        content: str
    ) -> MessageImportance:
        """Auto-detect message importance based on content."""
        content_lower = content.lower()
        
        # System messages with main objective are critical
        if role == MessageRole.SYSTEM and content_lower.startswith(MAIN_OBJECTIVE_PREFIX):
            return MessageImportance.CRITICAL
        
        # System messages are high importance
        if role == MessageRole.SYSTEM:
            return MessageImportance.HIGH
        
        # User messages with certain keywords are high importance
        high_importance_keywords = [
            "important", "critical", "remember", "note that",
            "make sure", "do not", "never", "always"
        ]
        if any(keyword in content_lower for keyword in high_importance_keywords):
            return MessageImportance.HIGH
        
        # Tool calls are medium importance
        if role in (MessageRole.TOOL, MessageRole.FUNCTION):
            return MessageImportance.MEDIUM
        
        # Short messages might be low importance
        if len(content.split()) < 5:
            return MessageImportance.LOW
        
        return MessageImportance.MEDIUM
    
    def _default_token_counter(self, text: str) -> int:
        """Default token counting (rough estimation)."""
        # Rough estimation: ~4 characters per token
        return max(1, len(text) // 4)


# Global context manager instance
_default_context_manager: Optional[ContextWindowManager] = None


def get_context_manager(
    max_tokens: int = 8000,
    strategy: str = "sliding"
) -> ContextWindowManager:
    """Get or create a context manager instance."""
    global _default_context_manager
    
    if _default_context_manager is None:
        _default_context_manager = ContextWindowManager(
            max_tokens=max_tokens,
            strategy=strategy
        )
    
    return _default_context_manager


def create_context_manager(
    max_tokens: int = 8000,
    strategy: Union[str, ContextStrategy] = "sliding",
    **kwargs
) -> ContextWindowManager:
    """Create a new context manager instance."""
    return ContextWindowManager(
        max_tokens=max_tokens,
        strategy=strategy,
        **kwargs
    )
