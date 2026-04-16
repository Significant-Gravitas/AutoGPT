"""Tool for storing memories in the Graphiti temporal knowledge graph."""

import logging
from typing import Any

from backend.copilot.graphiti.config import is_enabled_for_user
from backend.copilot.graphiti.ingest import enqueue_episode
from backend.copilot.graphiti.memory_model import (
    MemoryEnvelope,
    MemoryKind,
    MemoryStatus,
    ProcedureMemory,
    ProcedureStep,
    RuleMemory,
    SourceKind,
)
from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import ErrorResponse, MemoryStoreResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class MemoryStoreTool(BaseTool):
    """Store a memory/fact in the user's temporal knowledge graph."""

    @property
    def name(self) -> str:
        return "memory_store"

    @property
    def description(self) -> str:
        return (
            "Store a memory or fact about the user for future recall. "
            "Use when the user shares preferences, business context, decisions, "
            "relationships, or other important information worth remembering "
            "across sessions. Supports optional metadata for scoping and classification."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Brief descriptive name for this memory (e.g. 'user_prefers_python')",
                },
                "content": {
                    "type": "string",
                    "description": "The information to remember. Be concise but complete.",
                },
                "source_description": {
                    "type": "string",
                    "description": "Context about where this info came from",
                    "default": "Conversation memory",
                },
                "source_kind": {
                    "type": "string",
                    "enum": [e.value for e in SourceKind],
                    "description": "Who asserted this: user_asserted (default), assistant_derived, or tool_observed",
                    "default": "user_asserted",
                },
                "scope": {
                    "type": "string",
                    "description": "Namespace for this memory: 'real:global' (default), 'project:<name>', 'book:<title>'",
                    "default": "real:global",
                },
                "memory_kind": {
                    "type": "string",
                    "enum": [e.value for e in MemoryKind],
                    "description": "Type of memory: fact (default), preference, rule, finding, plan, event, procedure",
                    "default": "fact",
                },
                "rule": {
                    "type": "object",
                    "description": (
                        "Structured rule data — use when memory_kind=rule to preserve "
                        "exact operational instructions. Example: "
                        '{"instruction": "CC Sarah on client communications", '
                        '"actor": "Sarah", "trigger": "client-related communications"}'
                    ),
                    "properties": {
                        "instruction": {
                            "type": "string",
                            "description": "The actionable instruction",
                        },
                        "actor": {
                            "type": "string",
                            "description": "Who performs or is subject to the rule",
                        },
                        "trigger": {
                            "type": "string",
                            "description": "When the rule applies",
                        },
                        "negation": {
                            "type": "string",
                            "description": "What NOT to do, if applicable",
                        },
                    },
                    "required": ["instruction"],
                },
                "procedure": {
                    "type": "object",
                    "description": (
                        "Structured procedure data — use when memory_kind=procedure "
                        "for multi-step workflows with ordering, tools, and conditions."
                    ),
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "What this procedure accomplishes",
                        },
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "order": {
                                        "type": "integer",
                                        "description": "Step number",
                                    },
                                    "action": {
                                        "type": "string",
                                        "description": "What to do",
                                    },
                                    "tool": {
                                        "type": "string",
                                        "description": "Tool or service to use",
                                    },
                                    "condition": {
                                        "type": "string",
                                        "description": "When this step applies",
                                    },
                                    "negation": {
                                        "type": "string",
                                        "description": "What NOT to do",
                                    },
                                },
                                "required": ["order", "action"],
                            },
                        },
                    },
                    "required": ["description", "steps"],
                },
            },
            "required": ["name", "content"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        name: str = "",
        content: str = "",
        source_description: str = "Conversation memory",
        source_kind: str = "user_asserted",
        scope: str = "real:global",
        memory_kind: str = "fact",
        rule: dict | None = None,
        procedure: dict | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        if not user_id:
            return ErrorResponse(
                message="Authentication required to store memories.",
                session_id=session.session_id,
            )

        if not await is_enabled_for_user(user_id):
            return ErrorResponse(
                message="Memory features are not enabled for your account.",
                session_id=session.session_id,
            )

        if not name or not content:
            return ErrorResponse(
                message="Both 'name' and 'content' are required.",
                session_id=session.session_id,
            )

        rule_model = None
        if rule and memory_kind == "rule":
            try:
                rule_model = RuleMemory(**rule)
            except Exception:
                logger.warning("Invalid rule data, storing as plain fact")
                memory_kind = "fact"

        procedure_model = None
        if procedure and memory_kind == "procedure":
            try:
                steps = [ProcedureStep(**s) for s in procedure.get("steps", [])]
                procedure_model = ProcedureMemory(
                    description=procedure.get("description", content),
                    steps=steps,
                )
            except Exception:
                logger.warning("Invalid procedure data, storing as plain fact")
                memory_kind = "fact"

        try:
            resolved_source = SourceKind(source_kind)
        except ValueError:
            resolved_source = SourceKind.user_asserted
        try:
            resolved_kind = MemoryKind(memory_kind)
        except ValueError:
            resolved_kind = MemoryKind.fact

        envelope = MemoryEnvelope(
            content=content,
            source_kind=resolved_source,
            scope=scope,
            memory_kind=resolved_kind,
            status=MemoryStatus.active,
            provenance=session.session_id,
            rule=rule_model,
            procedure=procedure_model,
        )

        queued = await enqueue_episode(
            user_id,
            session.session_id,
            name=name,
            episode_body=envelope.model_dump_json(),
            source_description=source_description,
            is_json=True,
        )

        if not queued:
            return ErrorResponse(
                message="Memory queue is full — please try again shortly.",
                session_id=session.session_id,
            )

        return MemoryStoreResponse(
            message=f"Memory '{name}' queued for storage.",
            session_id=session.session_id,
            memory_name=name,
        )
