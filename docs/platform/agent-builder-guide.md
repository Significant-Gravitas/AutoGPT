# Agent Builder Guide

## Overview

The Agent Builder is the visual editor where you design and build agents by connecting blocks together on a canvas. Each block is a single action — like generating text, calling an API, or processing data — and you wire them together to create automated workflows.

**URL:** [platform.agpt.co/build](https://platform.agpt.co/build)

## The Builder Interface

When you open the builder, you'll see:

- **Canvas**: The main workspace where you place and connect blocks
- **Blocks Menu**: A panel on the left-hand side where you browse and search for blocks
- **Save Button**: Save your agent with a title and description

## Working with Blocks

### Types of Blocks

Blocks fall into three categories:

| Type | Description |
|------|-------------|
| **Input Blocks** | Define what information the agent needs when it runs. These become the input fields users fill in when starting a task. Types include text inputs, file inputs, and more. |
| **Action Blocks** | Perform operations — AI text generation, image creation, API calls, data processing, and hundreds of integrations with external platforms. |
| **Output Blocks** | Define what the agent returns as its result. These become the visible output when a task completes. |

Input and output blocks define the **schema** of your agent — they determine what users see when running the agent. All other blocks inside the agent are internal and not exposed to the user at runtime.

{% hint style="info" %}
There is also a special type of input block: **Trigger Blocks**. These allow your agent to be activated by external events via webhooks rather than manual input. See [Scheduling & Triggers](scheduling-and-triggers.md) for details.
{% endhint %}

### Adding Blocks

1. Open the **Blocks menu** on the left-hand side of the builder
2. Browse categories or use the search bar to find a specific block
3. Click on a block to add it to the canvas

There are hundreds of blocks available, integrating with many platforms and services.

### Connecting Blocks

Blocks have **input pins** and **output pins**. Pins are typed — they handle specific data types like text, numbers, files, and more.

To connect blocks:

1. Click on an **output pin** of one block
2. Drag to an **input pin** of another block (or simply click the output pin, then click the input pin)
3. A **connection line** will appear between the two pins

When the agent runs, data flows along these connections. This is visually represented by a **coloured bead** that slides along the connection line from the output pin to the input pin.

### Configuring Blocks

Many blocks have settings you can configure directly on the block. For example:

- The **AI Text Generator** block lets you choose which language model to use
- **Integration blocks** may require credentials (see [Integrations & Credentials](integrations-and-credentials.md))
- Some blocks allow you to hardcode values on their input pins instead of connecting them to other blocks

If a block requires a credential you haven't connected yet, a **credential bar** will appear prompting you to add it (via OAuth, API key, or username/password depending on the service).

## Saving Your Agent

To save your agent:

- Press **Ctrl+S**, or
- Click the **Save button** in the builder

When saving, you can provide a **title** and **description** for the agent. This is a local save to your personal library. For publishing to the public marketplace, see [Publishing to the Marketplace](marketplace.md#publishing-an-agent).

{% hint style="info" %}
There is currently no draft vs. saved state — saving an agent immediately updates it in your library.
{% endhint %}

## Navigating Back to Your Library

After saving, click the **Agents** button in the navigation bar to return to your library and find your agent.

## Editing an Existing Agent

To open an existing agent in the builder:

1. Go to your **Agent Library** (click **Agents** in the nav bar)
2. Click on the agent you want to edit
3. Click the **three dots** menu (⋯) on the far right-hand side of the screen
4. Select **Edit Agent**

This will open the agent in the builder with all its existing blocks and connections.

## Error Handling

When a block fails during execution, it produces data on its **error pin**. As the agent creator, you decide how to handle errors:

- **Surface the error**: Connect the error pin to an output block so the error is returned as the agent's result
- **Handle gracefully**: Connect the error pin to other blocks that provide fallback behaviour, ensuring the agent continues working even when individual blocks encounter problems

## Tips

- **Name your input and output blocks clearly** — these names become the labels users see when running your agent
- **Test incrementally** — save and run your agent frequently as you build to catch issues early
- **Use the search** in the blocks menu to quickly find what you need among hundreds of available blocks
- **Check the pin types** — connections only work between compatible pin types
