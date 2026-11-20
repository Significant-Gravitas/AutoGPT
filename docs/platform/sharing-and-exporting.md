# Sharing & Exporting Agents

## Overview

There are several ways to share agents and their outputs with others on the AutoGPT Platform. This guide covers all the available options.

## Sharing Options

### Share Task Output via URL

A completed task remains private until you explicitly enable sharing:

1. Open the agent and select the completed task.
2. Click **Share results**.
3. Review the public-access warning and click **Enable Sharing**.
4. Copy the generated **Share URL**.

Anyone with that URL can view the run's outputs. Use **Stop Sharing** in the same dialog to revoke the link.

{% hint style="warning" %}
A share URL is publicly accessible to anyone who has it. It does not publish the task inputs, but outputs may contain sensitive data.
{% endhint %}

### Publish to the Marketplace

The most visible way to share an agent is to publish it to the [Marketplace](marketplace.md). Published agents are discoverable by all platform users and go through a review process by the AutoGPT team.

See [Publishing an Agent](marketplace.md#publishing-an-agent) for the full guide.

### Export as a File

For sharing agents privately — without publishing to the marketplace — you can export an agent as a file:

1. Go to your [Agent Library](agent-library.md) and open the agent
2. Click the **three dots** (⋯) on the far right
3. Select **Export Agent to File**
4. The agent file will download to your computer

You can then send this file to anyone via email, messaging, or any other method.

### Import a Shared File

To import an agent file someone has shared with you:

1. Go to your [Agent Library](agent-library.md)
2. Click **Import**
3. Select the **AutoGPT agent** tab
4. Upload the exported `.json` file
5. The agent will be added to your library

## Teams & Collaboration

Organization and team workspaces are available only when enabled for your account or deployment. Use their built-in access controls where available. Export/import remains the portable way to transfer an agent file between separate accounts or deployments.
