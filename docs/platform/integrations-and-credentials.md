# Integrations & Credentials

## Overview

Many blocks on the AutoGPT Platform integrate with external services like Google, GitHub, Linear, Twitter, and more. These integrations require credentials — such as OAuth connections, API keys, or username/password pairs — to access your accounts on those services.

This guide explains how credentials work on the platform, how to add them, and how to manage them.

## How Credentials Work

### Platform-Provided Credentials

On the cloud-hosted platform at [platform.agpt.co](https://platform.agpt.co), many credentials are **provided by default**. Services like OpenAI, Anthropic, and Replicate are pre-configured, so you can use AI blocks and many other features without providing your own API keys.

### User-Provided Credentials

For services tied to your personal accounts — such as Google, Linear, GitHub, or Twitter — you'll need to connect your own credentials. This only needs to be done **once per service per account**. After connecting, all agents that use that service will automatically have access.

## Adding Credentials

Credentials are added **in context** — when you encounter a block that needs them, rather than from a central setup page.

### When Building an Agent

If you add a block to the builder that requires a credential you haven't connected yet, a **credential bar** will appear on the block prompting you to add it.

### When Running an Agent

If you run an agent (including marketplace agents) that requires credentials, one of the input fields will be the credential selector. This only appears for services you haven't connected yet.

### Credential Types

Depending on the service, you'll be prompted to authenticate in one of three ways:

| Type | Description | Example Services |
|------|-------------|------------------|
| **OAuth** | Click to authorise via the service's login page | Google, GitHub, Twitter |
| **API Key** | Paste your API key from the service's dashboard | Linear, OpenAI (if self-hosting) |
| **Username & Password** | Enter your account credentials | Varies by service |

{% hint style="info" %}
You only need to connect a credential **once per service**. For example, after adding your Linear API key, every agent that uses Linear blocks will have access automatically.
{% endhint %}

## Managing Credentials

### Viewing Connected Integrations

To view and manage your connected integrations:

1. Click your **profile picture** in the top-right corner
2. Select **Integrations**
3. You'll see a list of all your connected integrations

**URL:** [platform.agpt.co/profile/integrations](https://platform.agpt.co/profile/integrations)

### Removing a Credential

From the integrations page, you can **browse** your list of connected integrations and **delete** any you no longer need.

{% hint style="warning" %}
You **cannot add** new integrations from the integrations management screen. New credentials are only added when you encounter a block or agent that requires them.
{% endhint %}

## Self-Hosted Credentials

If you're running the platform locally via self-hosting, you'll need to provide your own API keys for all services, including AI providers. These are configured in the `autogpt_platform/backend/.env` file. See the [Self-Hosting Guide](getting-started.md) for details.
