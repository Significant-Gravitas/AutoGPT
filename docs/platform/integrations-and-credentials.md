# Integrations & Credentials

## Overview

Many blocks on the AutoGPT Platform integrate with external services like Google, GitHub, Linear, Twitter, and more. These integrations require credentials — such as OAuth connections, API keys, or username/password pairs — to access your accounts on those services.

This guide explains how credentials work on the platform, how to add them, and how to manage them.

## How Credentials Work

### Platform-Provided Credentials

On the cloud-hosted platform at [platform.agpt.co](https://platform.agpt.co), some plans and account configurations include managed AI providers. If a block requires credentials that are not included, the platform will prompt you to connect your own. See the [pricing page](https://agpt.co/pricing) for current plan details.

### User-Provided Credentials

For services tied to your personal accounts, connect your own credentials. Credentials can be reused across agents, but you choose which compatible credential an agent or task uses. Managed credentials may be selected automatically when available.

## Adding Credentials

You can connect credentials centrally from **Settings → Integrations → Connect Service**, or in context when a builder block or task asks for one.

### When Building an Agent

If a block needs credentials, select an existing compatible credential or connect a new one from the block.

### When Running an Agent

Credential fields show the credentials that task will use. Select an existing compatible credential or connect a new one.

### Credential Types

Depending on the service, you'll be prompted to authenticate in one of three ways:

| Type | Description | Example Services |
|------|-------------|------------------|
| **OAuth** | Click to authorise via the service's login page | Google, GitHub, Twitter |
| **API Key** | Paste your API key from the service's dashboard | Linear, OpenAI |
| **Username & Password** | Enter your account credentials | Varies by service |

{% hint style="info" %}
Credentials are reusable, but each block or task retains its selected credential reference.
{% endhint %}

## Connecting MCP Services

MCP services give agents access to remote tools, including public documentation, research resources, and your connected accounts. Open **Settings → Integrations**, search **Available integrations**, and select a service with the **MCP** badge. You can also find these services in **Connect Service**.

Read the service's purpose and setup requirements before connecting:

- **Public tools:** Choose **No sign-in**, when offered, and select **Check connection** to discover available tools. This check does not use saved account credentials or create an account connection.
- **Account tools:** Use one of the offered sign-in, API token, or Basic authentication options. The required key type, subscription, and administrator approval vary by service.
- **Regions and custom endpoints:** A **Setup required** badge means you must choose the correct region or supply a documented remote server URL for your tenant, deployment, or product. Follow the linked official setup guide.
- **Permissions:** Some services start with read permissions and offer **Allow changes** for additional actions. Others use the provider's consent screen or account roles. Review the permissions requested; MCP connections are not universally read-only.

Local desktop processes and unsupported authentication flows are hidden from the MCP catalog. A native integration for the same service may still be available. See the [hosted MCP catalog](contributing/hosted-mcp-catalog.md) for the service matrix, setup requirements, verification limits, and maintainer guidance.

## Managing Credentials

1. Open **Settings**.
2. Select **Integrations**.
3. Use **Connect Service** to add a credential, or manage/remove credentials already listed.

**URL:** [platform.agpt.co/settings/integrations](https://platform.agpt.co/settings/integrations)

{% hint style="warning" %}
Removing a credential can break agents, workflows, or active webhooks that reference it. Review the dependency warning before confirming removal.
{% endhint %}

## Self-Hosted Credentials

When self-hosting, configure deployment-level provider keys and OAuth application credentials in `autogpt_platform/backend/.env`, then connect end-user accounts from **Settings → Integrations** as needed. See the [Self-Hosting Guide](getting-started.md) for details.
