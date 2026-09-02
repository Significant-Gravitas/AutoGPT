"""Unit tests for the pure row→model helpers in mapping.py."""

import json

import prisma.models

from .mapping import credentials_from_nodes


def _node(node_id: str, constant_input: object) -> prisma.models.AgentNode:
    # Prisma Json fields validate from raw JSON text, the same way rows
    # arrive off the wire.
    return prisma.models.AgentNode(
        id=node_id,
        agentBlockId="block-1",
        agentGraphId="graph-1",
        agentGraphVersion=1,
        constantInput=json.dumps(constant_input),
        webhookId=None,
        metadata="{}",
    )


def test_credentials_from_nodes_extracts_and_dedupes():
    nodes = [
        _node(
            "node-1",
            {
                "credentials": {
                    "id": "cred-1",
                    "provider": "openai",
                    "title": "My OpenAI key",
                    "type": "api_key",
                },
                "prompt": "hello",
            },
        ),
        _node(
            "node-2",
            {
                "slack_credentials": {
                    "id": "cred-2",
                    "provider": "slack",
                    "type": "oauth2",
                }
            },
        ),
        # Same credential on a second node must not repeat in the card.
        _node(
            "node-3",
            {"credentials": {"id": "cred-1", "provider": "openai"}},
        ),
    ]

    creds = credentials_from_nodes(nodes)

    assert [(cred.id, cred.provider, cred.title) for cred in creds] == [
        ("cred-1", "openai", "My OpenAI key"),
        ("cred-2", "slack", None),
    ]


def test_credentials_from_nodes_skips_malformed_input():
    nodes = [
        # Legacy/hand-edited rows: non-dict blob, non-credential field with a
        # credential-shaped value, and a credential entry without an id.
        _node("node-1", "not-a-dict"),
        _node("node-2", {"settings": {"id": "cred-9", "provider": "github"}}),
        _node("node-3", {"credentials": {"provider": "github"}}),
        _node("node-4", {"credentials": "oops"}),
    ]

    assert credentials_from_nodes(nodes) == []
