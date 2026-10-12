# Maintaining MCP Integrations

[`mcp_catalog.json`](../../autogpt_platform/backend/backend/integrations/mcp_catalog.json) defines the official services shown in Integrations and the AutoPilot MCP guide. [`mcp_catalog.py`](../../autogpt_platform/backend/backend/integrations/mcp_catalog.py) validates the records and exposes them through `get_mcp_catalog()`.

## Add or update a service

- Keep a stable `mcp_...` ID and link directly to the vendor's setup documentation.
- Describe the server's purpose and required plan, admin approval, region, endpoint, and credentials in `description` and `setup_instructions`.
- Choose the connection setup that matches the vendor's server; see [Choose how users connect](#choose-how-users-connect).
- Order the supported `auth_methods` by preference; the first method is selected initially. Choices are `oauth`, `bearer`, `basic`, and `none`.
- Set `oauth_server_url` when the vendor supplies a separate signed-in endpoint.
- Set documented `oauth_scopes` for the initial grant and separate `oauth_write_scopes` for the optional **Allow changes** choice. An empty default list omits the scope parameter; omission/null uses discovery defaults.

## Choose how users connect

Match the entry to where the vendor runs its MCP server. Every URL must be a public HTTPS URL.

- **One public URL for every customer:** use `connection_mode: hosted` with `server_url`. The connect dialog shows the URL locked, and AutoPilot uses it. Examples: PostHog, Notion.
- **One public URL, and the service can also be self-hosted:** use `hosted` with `server_url` and `allow_custom_url: true`. The dialog opens on `server_url`, and users can replace it with their own instance's URL. AutoPilot still uses `server_url`. Say in `setup_instructions` how to connect a self-hosted instance, including any sign-in setup it needs. Example: OpenSEO.
- **A separate URL per region:** use `connection_mode: custom` with one `server_url_options` entry per region and no `server_url`. Users pick their region, and the picker's **Custom URL** option covers self-hosted instances. Prefer this to one entry per region. Don't make it a hosted entry that defaults to one region, because users in the other regions would connect to the wrong one. Examples: Amplitude, Langfuse.
- **A URL unique to each customer account:** use `connection_mode: custom` with no `server_url` and no options. Say in `setup_instructions` where users find their URL. The dialog opens with an empty URL field. Examples: Chargebee, Sourcegraph.

Both `custom` setups show **Setup required** in the integrations list. AutoPilot tells users to add their URL under Settings → Integrations before it can use the server.

## Validate the change

Check the vendor's current endpoint and authentication contract. Verify initialization and tool listing with the MCP client, then a benign read using the intended permissions. For OAuth, check registration, consent, and refresh with the deployment's callback and an authorized test account.

Update affected catalog, provider, guide, and page integration tests. Cover changes to region selection, custom URLs, token handling, or optional grants. Regenerate API types when the metadata schema changes, and run the relevant backend and frontend checks.
