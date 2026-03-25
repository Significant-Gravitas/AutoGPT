# Credits & Billing

## Overview

The AutoGPT Platform uses a credit system to manage usage. Credits are consumed when blocks execute during agent runs. This guide explains how credits work, how pricing is determined, and how to monitor your spending.

{% hint style="info" %}
The platform is currently in a **pre-release closed beta**. Pricing is subject to change.
{% endhint %}

## How Credits Work

Credits are consumed on a **per-block-run** basis. Each time a block executes during an agent run, it costs a certain number of credits. The price of a block covers its compute, development, and operational costs — there are no separate charges for infrastructure or API usage.

### Block Pricing

Block prices vary depending on the block:

- **Fixed-price blocks**: Some blocks have a flat price regardless of how they are configured (e.g., basic data processing blocks)
- **Variable-price blocks**: Some blocks have a price that changes based on the settings you choose. For example, the **AI Text Generator** block's price changes depending on which large language model you select

{% hint style="info" %}
The current pricing system charges a flat rate per model for AI blocks — you are **not** charged per token.
{% endhint %}

Users are not charged for anything else on the platform beyond block execution. There are no subscription fees, storage fees, or platform access fees.

## Checking Your Balance

Your credit balance is displayed in the **top-right corner** of the screen at all times, visible from any page on the platform.

## Viewing Task Costs

To see how many credits a specific agent run consumed:

1. Go to your [Agent Library](agent-library.md)
2. Open the agent
3. Click on a completed task in the left-hand pane
4. The **total credit cost** for that task is displayed at the top of the task detail view

{% hint style="info" %}
There is no centralised ledger for browsing all credit spend across your account. Credit costs are available on a per-task basis within each agent.
{% endhint %}

## Running Out of Credits

There are no hard limits on usage beyond your credit balance. If your credit balance reaches zero:

- **Running agents will stop executing**
- **Scheduled agents will not run** until credits are replenished
- You will need to add more credits to continue using the platform

## Adding Credits

Credits can be added through the platform. Navigate to your profile settings to manage your credit balance.
