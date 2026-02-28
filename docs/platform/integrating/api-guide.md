# AutoGPT Platform External API Guide

The AutoGPT Platform provides an External API that allows you to programmatically interact with agents, blocks, the marketplace, and more.

## API Documentation

Full API documentation with interactive examples is available at:

- **Main**: [https://backend.agpt.co/external-api/docs](https://backend.agpt.co/external-api/docs)
- **v2 API**: [https://backend.agpt.co/external-api/v2/docs](https://backend.agpt.co/external-api/v2/docs)
- **v1 API**: [https://backend.agpt.co/external-api/v1/docs](https://backend.agpt.co/external-api/v1/docs)

The Swagger UI documentation includes all available endpoints, request/response schemas, and allows you to try out API calls directly.

**Recommendation**: New integrations should use the v2 API.

## Authentication Methods

The External API supports two authentication methods:

### 1. API Keys

API keys are the simplest way to authenticate. Generate an API key from your AutoGPT Platform account settings and include it in your requests using the `X-API-Key` header:

```bash
# List available blocks
curl -H "X-API-Key: YOUR_API_KEY" \
  https://backend.agpt.co/external-api/v1/blocks
```

API keys are ideal for:
- Server-to-server integrations
- Personal scripts and automation
- Backend services

### 2. OAuth 2.0 (Single Sign-On)

For applications that need to act on behalf of users, use OAuth 2.0. This allows users to authorize your application to access their AutoGPT resources.

To get started:

1. Register an OAuth application (contact platform administrator)
2. Implement the OAuth flow as described in the [OAuth Guide](oauth-guide.md)
3. Go through the OAuth flow to authorize your app and obtain an access token
4. Make API requests with the access token in the `Authorization: Bearer` header:

```bash
curl -H "Authorization: Bearer agpt_xt_..." \
  https://backend.agpt.co/external-api/v1/blocks
```

OAuth is ideal for:

- Third-party applications
- "Sign in with AutoGPT" (SSO, Single Sign-On) functionality
- Applications that need user-specific permissions

See the [OAuth Integration Guide](oauth-guide.md) for complete OAuth implementation details.

## Available Scopes

When creating API keys or using OAuth, request only the scopes your application needs.

### Core Scopes

| Scope | Description |
|-------|-------------|
| `IDENTITY` | Read user ID, e-mail, and timezone |
| `READ_GRAPH` | Read graph/agent definitions and versions |
| `WRITE_GRAPH` | Create, update, and delete graphs |
| `READ_BLOCK` | Read block definitions |
| `READ_STORE` | Access the agent marketplace |
| `WRITE_STORE` | Create, update, and delete marketplace submissions |
| `READ_LIBRARY` | List library agents and their runs |
| `RUN_AGENT` | Execute agents from your library |
| `READ_RUN` | List and get execution run details |
| `WRITE_RUN` | Stop and delete runs |
| `READ_RUN_REVIEW` | List pending human-in-the-loop reviews |
| `WRITE_RUN_REVIEW` | Submit human-in-the-loop review responses |
| `READ_SCHEDULE` | List execution schedules |
| `WRITE_SCHEDULE` | Create and delete schedules |
| `READ_CREDITS` | Get credit balance and transaction history |
| `READ_INTEGRATIONS` | List OAuth credentials |
| `UPLOAD_FILES` | Upload files for agent input |

### Legacy Scopes (v1 only)

| Scope | Description |
|-------|-------------|
| `EXECUTE_GRAPH` | Execute graphs directly (use `RUN_AGENT` in v2) |
| `EXECUTE_BLOCK` | Execute individual blocks |
| `USE_TOOLS` | Use chat tools via external API |
| `MANAGE_INTEGRATIONS` | Initiate and complete OAuth flows |
| `DELETE_INTEGRATIONS` | Delete OAuth credentials |

## Support

For issues or questions about API integration:

- Open an issue on [GitHub](https://github.com/Significant-Gravitas/AutoGPT)
- Check the [Swagger documentation](https://backend.agpt.co/external-api/docs)
