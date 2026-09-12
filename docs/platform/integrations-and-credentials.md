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
