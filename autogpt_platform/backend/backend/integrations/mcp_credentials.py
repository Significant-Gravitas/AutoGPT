from backend.data.model import OAuth2Credentials


def is_manual_mcp_credential(credentials: OAuth2Credentials) -> bool:
    metadata = credentials.metadata or {}
    return (
        credentials.refresh_token is None
        and not metadata.get("mcp_token_url")
        and not metadata.get("mcp_client_id")
    )
