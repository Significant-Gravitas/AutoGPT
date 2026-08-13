# Marketplace

## Overview

The AutoGPT Marketplace is a public library of agents created by the community and by the AutoGPT team. You can discover agents for a wide range of use cases, add them to your library with one click, and even publish your own agents for others to use.

**URL:** [platform.agpt.co/marketplace](https://platform.agpt.co/marketplace)

**Access:** Click **Marketplace** in the navigation bar.

## Browsing the Marketplace

The marketplace allows you to:

- **Search** for agents by name or keyword
- **Browse by category** to explore different use cases
- **View agent details** by clicking on any agent

### Agent Detail Page

Each agent in the marketplace has a dedicated page that includes:

- **Title and description** of what the agent does
- **Creator information** — who built the agent and a link to their other agents
- **Video** (when available) — a demonstration of the agent in action
- **Output showcase** (when available) — examples of the agent's results
- **Instructions** — guidance on how to run the agent and what to expect

## Adding an Agent to Your Library

When you find an agent you want to use:

1. Click on the agent to open its marketplace page
2. Click the **Add to Library** button

The agent appears in your [Agent Library](agent-library.md), where you can run or schedule it. Marketplace agents open read-only; duplicate one before editing.

{% hint style="info" %}
If you already have the agent in your library, the button changes to **See Runs**, which takes you directly to that agent in your library.
{% endhint %}

### Download for Self-Hosting

Marketplace detail pages also provide **Download here** under “Want to use this agent locally?” Download the agent JSON and import it into your local instance. See [Download & Import an Agent](download-agent-from-marketplace-local.md) for details.

## Publishing an Agent

You can publish your own agents to the marketplace for the community to discover and use.

### How to Publish

1. Open the agent in the Builder and select **Publish to Marketplace**, or open **Settings → Creator Dashboard** and select **Publish Agent**
2. Choose the agent and version you want to publish if prompted
3. Fill out the publishing form:

| Field | Description |
|-------|-------------|
| **Title** | The name of your agent as it will appear in the marketplace |
| **Subheader** | A short tagline for your agent |
| **Slug** | A URL-friendly identifier (e.g., `resume-rewriter`) |
| **Thumbnail Images** | Upload images — the first image becomes the marketplace thumbnail |
| **YouTube Video Link** | Optional — a video demonstrating your agent |
| **Output demo** | Optional — a URL showing example output the agent produces |
| **Category** | Select the most relevant category for your agent |
| **Description** | A detailed explanation of what your agent does |
| **Instructions** | Explain to users how to run this agent and what to expect |
| **Recommended Schedule** | Suggest when users should run this agent for best results |

4. Click **Submit for review**

### Review Process

After submission, a member of the AutoGPT team will review your agent against curation standards. If approved, your agent will be published to the marketplace and visible to all users.

## Editing Marketplace Agents

An agent added from the marketplace is read-only. Open it in the builder and click **Duplicate** to create an editable fork in your library. Changes to that fork do not modify the original marketplace listing.
