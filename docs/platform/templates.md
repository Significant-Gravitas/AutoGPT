# Templates

## Overview

Templates save a completed task's inputs and selected credential references so you can start another task with the same configuration.

## Creating a Template

Templates are created from previously completed tasks:

1. Go to your [Agent Library](agent-library.md) and open the agent
2. Find a completed task in the left-hand pane whose inputs you want to reuse
3. Click the **three dots** (⋯) on the task
4. Select **Save as Template**
5. Give the template a **name** and **description**
6. Click **Save**

The template stores the task's input values, selected credential references, graph ID, and graph version. It does not copy credential secrets.

## Using a Template

To run an agent using a saved template:

1. Open the agent from your library
2. Switch to the **Templates** tab on the left-hand side
3. Click the template you want to use
4. Review or edit its saved inputs and credential selections
5. Select **Start task from template**

## Managing Templates

Templates are listed under the **Templates** tab on your agent's detail page. The tab shows the count of available templates (e.g., `Templates 3`).

## When to Use Templates

Templates are particularly useful for:

- **Recurring tasks**: When you run an agent regularly with the same inputs (e.g., a daily SEO report for the same keyword)
- **Standard configurations**: When you've found input settings that work well and want to reuse them
- **Quick re-runs**: When you want to repeat a successful task without re-entering all the inputs

{% hint style="info" %}
A template is pinned to the agent graph version from which it was created. Later agent edits do not automatically update the template to a newer graph version.
{% endhint %}
