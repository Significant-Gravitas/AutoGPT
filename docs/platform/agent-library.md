# Agent Library

## Overview

The Agent Library is your personal collection of agents. From here you can run agents, view task history, set up schedules and triggers, edit agents, and manage your collection.

**URL:** [platform.agpt.co/library](https://platform.agpt.co/library)

**Access:** Click the **Agents** button in the navigation bar.

## Library View

Your library displays all of your agents, including agents you've built yourself and agents you've added from the marketplace. Each agent shows its name and key information at a glance.

### Favouriting Agents

To keep important agents easy to find, click the **heart icon** on any agent to favourite it. Favourited agents are pinned to the top of your library.

{% hint style="info" %}
There is currently no folder system for organising agents. Use favourites to pin your most-used agents to the top.
{% endhint %}

## Agent Detail View

Click on any agent to open its detail view. This screen provides:

- **Tasks**: A full history of every time this agent has been executed
- **Scheduled**: A list of active schedules for this agent
- **Templates**: Saved input configurations for quick re-runs

These are accessible via tabs on the left-hand side of the agent screen (e.g., `Tasks 286 | Scheduled 1 | Templates 0`).

### Agent Actions Menu

Click the **three dots** (⋯) on the far right-hand side of the agent screen to access:

| Action | Description |
|--------|-------------|
| **Edit Agent** | Opens the agent in the builder for editing |
| **Delete Agent** | Permanently removes the agent from your library |
| **Export Agent to File** | Downloads the agent as a file you can share with others |

## Running an Agent

To run an agent manually:

1. Open the agent from your library
2. Click **New Task**
3. Fill in the required input fields
4. Click **Start Task**

The task will begin executing immediately. You can watch the progress and view the results once it completes.

{% hint style="warning" %}
If the agent uses **trigger blocks** instead of standard input blocks, the **New Task** button is replaced with **New Trigger**. See [Scheduling & Triggers](scheduling-and-triggers.md) for details.
{% endhint %}

## Viewing Task Results

Every time an agent runs, it creates a **task**. To view task results:

1. Open the agent from your library
2. In the left-hand pane, browse the list of completed tasks
3. Click on a task to view its details

A task detail view shows:

- **Inputs**: The values that were provided when the task started
- **Outputs**: The results the agent produced
- **Cost**: The total credit cost for this task execution, displayed at the top of the task

You can also **share a task** by copying its URL, which allows others to view the task output directly.

## Uploading an Agent

To import an agent from a file:

1. Go to your Agent Library
2. Click **Upload Agent** at the top
3. Select the agent file from your computer

The agent will be added to your library and can be run or edited like any other agent. This is useful for importing agents shared by other users outside of the marketplace.

## Deleting an Agent

1. Open the agent from your library
2. Click the **three dots** (⋯) on the far right
3. Select **Delete Agent**
4. Confirm the deletion when prompted

{% hint style="danger" %}
Deleting an agent is permanent and cannot be undone. Make sure you want to remove it before confirming.
{% endhint %}
