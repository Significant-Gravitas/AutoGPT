# Development Setup for Lighthouse CI

This document outlines the required setup steps to ensure the frontend builds properly with the Lighthouse CI integration.

## Required Steps Before Building

### 1. Start Backend Services

The frontend build requires generated API files that depend on the running backend. Start the services:

```bash
cd autogpt_platform
docker compose --profile local up deps --build --detach
docker compose up rest_server executor websocket_server database_manager scheduler_server notification_server -d
```

### 2. Generate API Files

Once the backend is running, generate the TypeScript API client files:

```bash
cd autogpt_platform/frontend
pnpm generate:api:force
```

This creates:

- `src/app/api/__generated__/endpoints/` - API endpoint hooks
- `src/app/api/__generated__/models/` - TypeScript type definitions
- `src/app/api/openapi.json` - OpenAPI specification

### 3. Verify Setup

Test that TypeScript compilation works:

```bash
pnpm types
```

## Why This is Needed

The frontend application uses auto-generated TypeScript clients that are created from the backend's OpenAPI specification. These files are:

1. **Not committed to git** (they're in .gitignore)
2. **Generated at build time** in CI environments
3. **Required for TypeScript compilation** to succeed

## CI Environment

In CI, this process is handled automatically by the `generate:api:force` command in the build pipeline. The Docker services are already running as part of the test environment.

## Common Issues

- **Module not found errors**: Usually indicates API files weren't generated
- **Build failures**: Often resolved by ensuring backend services are healthy
- **TypeScript errors**: Check that `pnpm generate:api:force` completed successfully

## Lighthouse Integration

With this setup complete, the Lighthouse CI integration will work correctly:

- Local testing: `pnpm lighthouse:local`
- Full audit: `pnpm lighthouse`
- CI integration: Automatic via GitHub Actions
