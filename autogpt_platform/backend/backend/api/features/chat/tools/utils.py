"""Shared utilities for chat tools."""

import logging
from typing import Any

from backend.api.features.library import db as library_db
from backend.api.features.library import model as library_model
from backend.api.features.store import db as store_db
from backend.data.graph import GraphModel
from backend.data.model import (
    Credentials,
    CredentialsFieldInfo,
    CredentialsMetaInput,
    HostScopedCredentials,
    OAuth2Credentials,
)
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
    graph = await store_db.get_available_graph(
        store_agent.store_listing_version_id, hide_nodes=False
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


def _serialize_missing_credential(
    field_key: str, field_info: CredentialsFieldInfo
) -> dict[str, Any]:
    """
    Convert credential field info into a serializable dict that preserves all supported
    credential types (e.g., api_key + oauth2) so the UI can offer multiple options.
    """
    supported_types = sorted(field_info.supported_types)
    provider = next(iter(field_info.provider), "unknown")
    scopes = sorted(field_info.required_scopes or [])

    return {
        "id": field_key,
        "title": field_key.replace("_", " ").title(),
        "provider": provider,
        "provider_name": provider.replace("_", " ").title(),
        "type": supported_types[0] if supported_types else "api_key",
        "types": supported_types,
        "scopes": scopes,
    }


def build_missing_credentials_from_graph(
    graph: GraphModel, matched_credentials: dict[str, CredentialsMetaInput] | None
) -> dict[str, Any]:
    """
    Build a missing_credentials mapping from a graph's aggregated credentials inputs,
    preserving all supported credential types for each field.
    """
    matched_keys = set(matched_credentials.keys()) if matched_credentials else set()
    aggregated_fields = graph.aggregate_credentials_inputs()

    return {
        field_key: _serialize_missing_credential(field_key, field_info)
        for field_key, (field_info, _, _) in aggregated_fields.items()
        if field_key not in matched_keys
    }


def build_missing_credentials_from_field_info(
    credential_fields: dict[str, CredentialsFieldInfo],
    matched_keys: set[str],
) -> dict[str, Any]:
    """
    Build missing_credentials mapping from a simple credentials field info dictionary.
    """
    return {
        field_key: _serialize_missing_credential(field_key, field_info)
        for field_key, field_info in credential_fields.items()
        if field_key not in matched_keys
    }


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


async def match_credentials_to_requirements(
    user_id: str,
    requirements: dict[str, CredentialsFieldInfo],
) -> tuple[dict[str, CredentialsMetaInput], list[CredentialsMetaInput]]:
    """
    Match user's credentials against a dictionary of credential requirements.

    This is the core matching logic shared by both graph and block credential matching.
    """
    matched: dict[str, CredentialsMetaInput] = {}
    missing: list[CredentialsMetaInput] = []

    if not requirements:
        return matched, missing

    available_creds = await get_user_credentials(user_id)

    for field_name, field_info in requirements.items():
        matching_cred = find_matching_credential(available_creds, field_info)

        if matching_cred:
            try:
                matched[field_name] = create_credential_meta_from_match(matching_cred)
            except Exception as e:
                logger.error(
                    f"Failed to create CredentialsMetaInput for field '{field_name}': "
                    f"provider={matching_cred.provider}, type={matching_cred.type}, "
                    f"credential_id={matching_cred.id}",
                    exc_info=True,
                )
                provider = next(iter(field_info.provider), "unknown")
                cred_type = next(iter(field_info.supported_types), "api_key")
                missing.append(
                    CredentialsMetaInput(
                        id=field_name,
                        provider=provider,  # type: ignore
                        type=cred_type,  # type: ignore
                        title=f"{field_name} (validation failed: {e})",
                    )
                )
        else:
            provider = next(iter(field_info.provider), "unknown")
            cred_type = next(iter(field_info.supported_types), "api_key")
            missing.append(
                CredentialsMetaInput(
                    id=field_name,
                    provider=provider,  # type: ignore
                    type=cred_type,  # type: ignore
                    title=field_name.replace("_", " ").title(),
                )
            )

    return matched, missing


async def get_user_credentials(user_id: str) -> list[Credentials]:
    """Get all available credentials for a user."""
    creds_manager = IntegrationCredentialsManager()
    return await creds_manager.store.get_all_creds(user_id)


def find_matching_credential(
    available_creds: list[Credentials],
    field_info: CredentialsFieldInfo,
) -> Credentials | None:
    """Find a credential that matches the required provider, type, scopes, and host."""
    for cred in available_creds:
        if cred.provider not in field_info.provider:
            continue
        if cred.type not in field_info.supported_types:
            continue
        if cred.type == "oauth2" and not _credential_has_required_scopes(
            cred, field_info
        ):
            continue
        if cred.type == "host_scoped" and not _credential_is_for_host(cred, field_info):
            continue
        return cred
    return None


def create_credential_meta_from_match(
    matching_cred: Credentials,
) -> CredentialsMetaInput:
    """Create a CredentialsMetaInput from a matched credential."""
    return CredentialsMetaInput(
        id=matching_cred.id,
        provider=matching_cred.provider,  # type: ignore
        type=matching_cred.type,
        title=matching_cred.title,
    )


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
        _,
        _,
    ) in aggregated_creds.items():
        # Find first matching credential by provider, type, and scopes
        matching_cred = next(
            (
                cred
                for cred in available_creds
                if cred.provider in credential_requirements.provider
                and cred.type in credential_requirements.supported_types
                and (
                    cred.type != "oauth2"
                    or _credential_has_required_scopes(cred, credential_requirements)
                )
                and (
                    cred.type != "host_scoped"
                    or _credential_is_for_host(cred, credential_requirements)
                )
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
            # Build a helpful error message including scope requirements
            error_parts = [
                f"provider in {list(credential_requirements.provider)}",
                f"type in {list(credential_requirements.supported_types)}",
            ]
            if credential_requirements.required_scopes:
                error_parts.append(
                    f"scopes including {list(credential_requirements.required_scopes)}"
                )
            missing_creds.append(
                f"{credential_field_name} (requires {', '.join(error_parts)})"
            )

    logger.info(
        f"Credential matching complete: {len(graph_credentials_inputs)}/{len(aggregated_creds)} matched"
    )

    return graph_credentials_inputs, missing_creds


def _credential_has_required_scopes(
    credential: OAuth2Credentials,
    requirements: CredentialsFieldInfo,
) -> bool:
    """Check if an OAuth2 credential has all the scopes required by the input."""
    # If no scopes are required, any credential matches
    if not requirements.required_scopes:
        return True
    return set(credential.scopes).issuperset(requirements.required_scopes)


def _credential_is_for_host(
    credential: HostScopedCredentials,
    requirements: CredentialsFieldInfo,
) -> bool:
    """Check if a host-scoped credential matches the host required by the input."""
    # We need to know the host to match host-scoped credentials to.
    # Graph.aggregate_credentials_inputs() adds the node's set URL value (if any)
    # to discriminator_values. No discriminator_values -> no host to match against.
    if not requirements.discriminator_values:
        return True

    # Check that credential host matches required host.
    # Host-scoped credential inputs are grouped by host, so any item from the set works.
    return credential.matches_url(list(requirements.discriminator_values)[0])


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
