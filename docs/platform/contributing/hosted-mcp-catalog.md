# Maintaining MCP Integrations

[`mcp_catalog.json`](../../../autogpt_platform/backend/backend/integrations/mcp_catalog.json) defines the official services shown in Integrations and the AutoPilot MCP guide. [`mcp_catalog.py`](../../../autogpt_platform/backend/backend/integrations/mcp_catalog.py) validates the records and exposes them through `get_mcp_catalog()`.

## Add or update a service

- Keep a stable `mcp_...` ID and link directly to the vendor's setup documentation.
- Describe the server's purpose and required plan, admin approval, region, endpoint, and credentials in `description` and `setup_instructions`.
- Use `connection_mode: hosted` with a fixed public HTTPS `server_url`, or `custom` for a user-provided endpoint. Add `server_url_options` for documented regions.
- Order the supported `auth_methods` by preference; the first method is selected initially. Choices are `oauth`, `bearer`, `basic`, and `none`.
- Set `oauth_server_url` when the vendor supplies a separate signed-in endpoint.
- Set documented `oauth_scopes` for the initial grant and separate `oauth_write_scopes` for the optional **Allow changes** choice. An empty default list omits the scope parameter; omission/null uses discovery defaults.

## Validate the change

Check the vendor's current endpoint and authentication contract. Verify initialization and tool listing with the MCP client, then a benign read using the intended permissions. For OAuth, check registration, consent, and refresh with the deployment's callback and an authorized test account.

Update affected catalog, provider, guide, and page integration tests. Cover changes to region selection, custom URLs, token handling, or optional grants. Regenerate API types when the metadata schema changes, and run the relevant backend and frontend checks.
