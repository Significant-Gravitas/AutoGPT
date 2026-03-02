# LLM Registry Admin Guide

The LLM Registry is a database-driven system that manages all available LLM models, providers, and creators across your AutoGPT platform. This allows platform administrators to dynamically control which models are available without requiring code deployments.

## Overview

The LLM Registry replaces the previous hardcoded model system with a flexible, admin-controlled registry accessible at `/admin/llms`. This enables you to:

- Add and configure LLM models from any provider
- Set model costs and pricing tiers
- Enable or disable models at runtime
- Migrate workflows to different models when needed
- Configure a platform-wide recommended default model

## Accessing the LLM Registry

Navigate to `/admin/llms` in your AutoGPT platform (admin privileges required). The admin UI has four main sections:

1. **Providers** — LLM providers (OpenAI, Anthropic, etc.)
2. **Models** — Individual LLM models with configurations
3. **Creators** — Organizations that created/trained models
4. **Migrations** — Model migration history and management

## Managing Providers

Providers represent the API endpoints for LLM services (e.g., OpenAI, Anthropic, Groq).

### Adding a Provider

1. Click "Add Provider" in the Providers tab
2. Fill in the provider details:
   - **Name**: Internal identifier (e.g., `openai`, `anthropic`)
   - **Display Name**: User-facing name
   - **Description**: Optional provider description
   - **Default Credentials**: Default credential provider name
   - **Capabilities**: Check which features the provider supports:
     - Tools/function calling
     - JSON output mode
     - Reasoning tokens
     - Parallel tool calls

### Editing a Provider

Click the edit icon next to any provider to update its configuration. Changes take effect immediately.

## Managing Models

Models are the individual LLM variants available for blocks to use.

### Adding a Model

1. Click "Add Model" in the Models tab
2. Configure the model:
   - **Slug**: Unique identifier (e.g., `gpt-4o`, `claude-opus-4-6`)
   - **Display Name**: User-facing name
   - **Provider**: Select from your configured providers
   - **Creator**: Select the organization that trained the model
   - **Context Window**: Maximum input tokens
   - **Max Output Tokens**: Maximum generation length
   - **Enabled**: Whether the model is available for use
   - **Costs**: Add cost entries for different credential providers and units (per-run or per-token)

### Model Costs

Each model can have multiple cost entries for different credential providers. For example:
- OpenAI API credentials: 5 credits per run
- Custom self-hosted deployment: 2 credits per run

Cost units can be:
- **RUN**: Flat cost per model invocation
- **TOKEN**: Cost scales with token usage

### Disabling a Model

You have two options when disabling a model:

#### Simple Disable
Just uncheck "Enabled" on the model edit form. Workflows using this model will fail until it's re-enabled.

#### Disable with Migration
Use the "Toggle Model" feature to disable a model AND migrate all existing workflows to a replacement:

1. Click "Toggle" on the model
2. Select the replacement model slug
3. Optionally set a custom credit cost for migrated workflows
4. Provide a migration reason (e.g., "Provider outage")

This creates a migration record and updates all `AgentNode` records using the old model.

## Model Migrations

The migration system tracks when workflows are moved from one model to another, allowing you to revert if needed.

### Viewing Migrations

The Migrations tab shows all model migration history with:
- Source and target model
- Number of nodes migrated
- Migration reason
- Timestamp
- Revert status

### Reverting a Migration

If the original model becomes available again:

1. Find the migration in the Migrations tab
2. Click "Revert"
3. Choose whether to re-enable the source model
4. All affected workflows are switched back to the original model

### Custom Pricing for Migrations

When migrating, you can set a custom credit cost that overrides the target model's normal cost. This is useful if you want to:
- Charge the same cost as the original model
- Offer discounted pricing during a migration period
- Apply special pricing for affected users

The custom cost is stored in the migration record and applied to all affected workflows.

## Recommended Model

The platform can have one recommended model that serves as the default for all LLM blocks when no model is explicitly specified.

### Setting the Recommended Model

1. Go to the Models tab
2. Use the "Recommended Model" selector at the top
3. Choose the model you want as the platform default
4. Save

All blocks with `model` inputs using `default_factory=LlmModel.default` will use this model.

### When to Change It

- After adding a better/cheaper model
- During provider outages or deprecations
- For platform-wide cost optimization

## Model Creators

Creators represent the organizations that developed and trained the models (e.g., OpenAI, Meta, Anthropic).

### Adding a Creator

1. Click "Add Creator"
2. Fill in:
   - **Name**: Internal identifier
   - **Display Name**: User-facing name
   - **Description**: Optional background information
   - **Website URL**: Link to creator's site
   - **Logo URL**: Optional brand logo

Creators are used for organization and attribution in the UI.

## How Block Defaults Work

When a block has a `model` input:

- **If user selects a model**: That model is used
- **If user leaves it blank**: The platform's recommended model is used
- **If a model is disabled**: Workflows fail (unless migrated)
- **If a fallback exists**: Same-provider fallback may be used automatically

## Common Admin Tasks

### Adding a New Provider and Models

1. Add the provider (e.g., "groq")
2. Add its creator if not already present (e.g., "Meta")
3. Add individual models from that provider
4. Configure costs for each model
5. Enable the models you want available
6. Optionally set one as the recommended default

### Responding to a Provider Outage

1. Go to the affected provider's models
2. For each enabled model, click "Toggle"
3. Select a replacement model from a different provider
4. Set migration reason: "Provider outage"
5. Optionally set custom cost to match original pricing
6. When the provider recovers, revert the migrations

### Deprecating Old Models

1. Disable the deprecated model with migration
2. Choose a newer/better replacement
3. Reason: "Model deprecated by provider"
4. Monitor costs and performance of the replacement
5. After validation period, keep the migration permanent

## Architecture Notes

- Models are stored in the `LlmModel` database table
- Costs are in `LlmModelCost` with foreign key to model
- Migrations are tracked in `LlmModelMigration`
- The registry is cached and refreshed automatically
- Changes take effect immediately without deployments

## Troubleshooting

### "Model not found in registry"
- Check that the model exists in `/admin/llms`
- Verify the slug matches exactly (case-sensitive)
- Refresh the registry if you just added it

### "Model exists but is disabled"
- Enable it in the model edit form, or
- Provide a migration target when disabling

### Workflows still using old model after migration
- Check the migration record was created
- Verify the migration wasn't reverted
- Inspect the `AgentNode` records in the database

## Best Practices

1. **Always provide a migration target when disabling models** — prevents workflow failures
2. **Use descriptive migration reasons** — helps with audit trails
3. **Test new models before setting as recommended** — validate cost/performance
4. **Keep at least one model from each major provider enabled** — redundancy during outages
5. **Review migration history periodically** — clean up old migrations if needed
6. **Document custom costs in migration reasons** — explain pricing changes

## Security Considerations

- Only platform admins should have access to `/admin/llms`
- Model costs directly affect user credits and billing
- Migration records are permanent for audit purposes
- Credential provider names must match exactly with the credentials system

---

For questions or issues with the LLM Registry, contact your platform administrator.
