"""Shared utilities for chat tools."""

import logging
from typing import Any

from backend.api.features.library import db as library_db
from backend.api.features.library import model as library_model
from backend.api.features.store import db as store_db
from backend.data import graph as graph_db
from backend.data.graph import GraphModel
from backend.data.model import CredentialsMetaInput
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.util.exceptions import NotFoundError

logger = logging.getLogger(__name__)


async def fetch_graph_from_store_slug(
    username: str,
    agent_name: str,
) -> tuple[GraphModel | None, Any | None]:
    """
    Fetch graph from store by username/agent_name slug.

    Args:
        username: Creator's username
        agent_name: Agent name/slug

    Returns:
        tuple[Graph | None, StoreAgentDetails | None]: The graph and store agent details,
        or (None, None) if not found.

    Raises:
        DatabaseError: If there's a database error during lookup.
    """
    try:
        store_agent = await store_db.get_store_agent_details(username, agent_name)
    except NotFoundError:
        return None, None

    # Get the graph from store listing version
    graph_meta = await store_db.get_available_graph(
        store_agent.store_listing_version_id
    )
    graph = await graph_db.get_graph(
        graph_id=graph_meta.id,
        version=graph_meta.version,
        user_id=None,  # Public access
        include_subgraphs=True,
    )
    return graph, store_agent


def extract_credentials_from_schema(
    credentials_input_schema: dict[str, Any] | None,
) -> list[CredentialsMetaInput]:
    """
    Extract credential requirements from graph's credentials_input_schema.

    This consolidates duplicated logic from get_agent_details.py and setup_agent.py.

    Args:
        credentials_input_schema: The credentials_input_schema from a Graph object

    Returns:
        List of CredentialsMetaInput with provider and type info
    """
    credentials: list[CredentialsMetaInput] = []

    if (
        not isinstance(credentials_input_schema, dict)
        or "properties" not in credentials_input_schema
    ):
        return credentials

    for cred_name, cred_schema in credentials_input_schema["properties"].items():
        provider = _extract_provider_from_schema(cred_schema)
        cred_type = _extract_credential_type_from_schema(cred_schema)

        credentials.append(
            CredentialsMetaInput(
                id=cred_name,
                title=cred_schema.get("title", cred_name),
                provider=provider,  # type: ignore
                type=cred_type,  # type: ignore
            )
        )

    return credentials


def extract_credentials_as_dict(
    credentials_input_schema: dict[str, Any] | None,
) -> dict[str, CredentialsMetaInput]:
    """
    Extract credential requirements as a dict keyed by field name.

    Args:
        credentials_input_schema: The credentials_input_schema from a Graph object

    Returns:
        Dict mapping field name to CredentialsMetaInput
    """
    credentials: dict[str, CredentialsMetaInput] = {}

    if (
        not isinstance(credentials_input_schema, dict)
        or "properties" not in credentials_input_schema
    ):
        return credentials

    for cred_name, cred_schema in credentials_input_schema["properties"].items():
        provider = _extract_provider_from_schema(cred_schema)
        cred_type = _extract_credential_type_from_schema(cred_schema)

        credentials[cred_name] = CredentialsMetaInput(
            id=cred_name,
            title=cred_schema.get("title", cred_name),
            provider=provider,  # type: ignore
            type=cred_type,  # type: ignore
        )

    return credentials


def _extract_provider_from_schema(cred_schema: dict[str, Any]) -> str:
    """Extract provider from credential schema."""
    if "credentials_provider" in cred_schema and cred_schema["credentials_provider"]:
        return cred_schema["credentials_provider"][0]
    if "properties" in cred_schema and "provider" in cred_schema["properties"]:
        return cred_schema["properties"]["provider"].get("const", "unknown")
    return "unknown"


def _extract_credential_type_from_schema(cred_schema: dict[str, Any]) -> str:
    """Extract credential type from credential schema."""
    if "credentials_types" in cred_schema and cred_schema["credentials_types"]:
        return cred_schema["credentials_types"][0]
    if "properties" in cred_schema and "type" in cred_schema["properties"]:
        return cred_schema["properties"]["type"].get("const", "api_key")
    return "api_key"


async def get_or_create_library_agent(
    graph: GraphModel,
    user_id: str,
) -> library_model.LibraryAgent:
    """
    Get existing library agent or create new one.

    This consolidates duplicated logic from run_agent.py and setup_agent.py.

    Args:
        graph: The Graph to add to library
        user_id: The user's ID

    Returns:
        LibraryAgent instance
    """
    existing = await library_db.get_library_agent_by_graph_id(
        graph_id=graph.id, user_id=user_id
    )
    if existing:
        return existing

    library_agents = await library_db.create_library_agent(
        graph=graph,
        user_id=user_id,
        create_library_agents_for_sub_graphs=False,
    )
    assert len(library_agents) == 1, "Expected 1 library agent to be created"
    return library_agents[0]


async def match_user_credentials_to_graph(
    user_id: str,
    graph: GraphModel,
) -> tuple[dict[str, CredentialsMetaInput], list[str]]:
    """
    Match user's available credentials against graph's required credentials.

    Uses graph.aggregate_credentials_inputs() which handles credentials from
    multiple nodes and uses frozensets for provider matching.

    Args:
        user_id: The user's ID
        graph: The Graph with credential requirements

    Returns:
        tuple[matched_credentials dict, missing_credential_descriptions list]
    """
    graph_credentials_inputs: dict[str, CredentialsMetaInput] = {}
    missing_creds: list[str] = []

    # Get aggregated credentials requirements from the graph
    aggregated_creds = graph.aggregate_credentials_inputs()
    logger.debug(
        f"Matching credentials for graph {graph.id}: {len(aggregated_creds)} required"
    )

    if not aggregated_creds:
        return graph_credentials_inputs, missing_creds

    # Get all available credentials for the user
    creds_manager = IntegrationCredentialsManager()
    available_creds = await creds_manager.store.get_all_creds(user_id)

    # For each required credential field, find a matching user credential
    # field_info.provider is a frozenset because aggregate_credentials_inputs()
    # combines requirements from multiple nodes. A credential matches if its
    # provider is in the set of acceptable providers.
    for credential_field_name, (
        credential_requirements,
        _node_fields,
    ) in aggregated_creds.items():
        # Find first matching credential by provider and type
        matching_cred = next(
            (
                cred
                for cred in available_creds
                if cred.provider in credential_requirements.provider
                and cred.type in credential_requirements.supported_types
            ),
            None,
        )

        if matching_cred:
            try:
                graph_credentials_inputs[credential_field_name] = CredentialsMetaInput(
                    id=matching_cred.id,
                    provider=matching_cred.provider,  # type: ignore
                    type=matching_cred.type,
                    title=matching_cred.title,
                )
            except Exception as e:
                logger.error(
                    f"Failed to create CredentialsMetaInput for field '{credential_field_name}': "
                    f"provider={matching_cred.provider}, type={matching_cred.type}, "
                    f"credential_id={matching_cred.id}",
                    exc_info=True,
                )
                missing_creds.append(
                    f"{credential_field_name} (validation failed: {e})"
                )
        else:
            missing_creds.append(
                f"{credential_field_name} "
                f"(requires provider in {list(credential_requirements.provider)}, "
                f"type in {list(credential_requirements.supported_types)})"
            )

    logger.info(
        f"Credential matching complete: {len(graph_credentials_inputs)}/{len(aggregated_creds)} matched"
    )

    return graph_credentials_inputs, missing_creds


async def check_user_has_required_credentials(
    user_id: str,
    required_credentials: list[CredentialsMetaInput],
) -> list[CredentialsMetaInput]:
    """
    Check which required credentials the user is missing.

    Args:
        user_id: The user's ID
        required_credentials: List of required credentials

    Returns:
        List of missing credentials (empty if user has all)
    """
    if not required_credentials:
        return []

    creds_manager = IntegrationCredentialsManager()
    available_creds = await creds_manager.store.get_all_creds(user_id)

    missing: list[CredentialsMetaInput] = []
    for required in required_credentials:
        has_matching = any(
            cred.provider == required.provider and cred.type == required.type
            for cred in available_creds
        )
        if not has_matching:
            missing.append(required)

    return missing
