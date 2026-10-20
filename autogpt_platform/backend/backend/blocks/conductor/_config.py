from backend.sdk import ProviderBuilder

# Conductor bills per seat, not per API call, so the provider registers no
# base_cost and its blocks report no provider_cost.
conductor = (
    ProviderBuilder("conductor")
    .with_description(
        "Cloud coding-agent workspaces: create workspaces, prompt agents, "
        "read transcripts and manage preview URLs, sections and routines"
    )
    .with_supported_auth_types("api_key")
    .build()
)
