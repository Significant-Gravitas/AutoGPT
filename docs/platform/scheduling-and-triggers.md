# Scheduling & Triggers

## Overview

AutoGPT agents can be run on demand, on a recurring schedule, or automatically in response to external events via webhooks. This guide covers both scheduling and trigger-based execution.

## Scheduling an Agent

Scheduling lets you run an agent automatically on a recurring basis with pre-configured inputs.

### Setting Up a Schedule

1. Go to your [Agent Library](agent-library.md) and open the agent you want to schedule
2. Click **New Task** (the same button used for manual runs)
3. Fill in all the input fields with the values you want the agent to use
4. At the bottom of the input form, you'll see two buttons: **Start Task** and **Schedule Task**
5. Click **Schedule Task**

### Configuring the Schedule

The schedule configuration screen allows you to set:

| Setting | Description |
|---------|-------------|
| **Schedule Name** | A descriptive name for this schedule |
| **Repeats** | Frequency — e.g., Weekly |
| **Repeats On** | Select specific days (individual days, weekdays, weekends, or select all) |
| **At** | The time of day to run (hour and minute) |
| **Timezone** | Schedule runs in your local timezone, displayed on screen |

**Example:** Run "Keyword SEO Expert" every weekday at 9:00 AM CST.

### Managing Schedules

After creating a schedule, you can view and manage it on your agent's detail page:

- Open the agent from your library
- Select the **Scheduled** tab on the left-hand side
- The tab shows the count of active schedules (e.g., `Scheduled 1`)

## Triggers (Webhook-Based Execution)

Triggers allow external services or your own code to start an agent automatically by sending data to a webhook URL.

### How Triggers Work

Unlike standard input blocks, **trigger blocks** are a special type of input block. When you add a trigger block to your agent in the builder, the agent's execution model changes — instead of being started manually, it waits for incoming webhook events.

### Setting Up a Trigger

#### Step 1: Add a Trigger Block in the Builder

1. Open the [Agent Builder](agent-builder-guide.md)
2. From the blocks menu, add a **trigger block** to your agent
3. Connect it to the rest of your workflow like any other input block
4. Save your agent

#### Step 2: Configure the Trigger in Your Library

1. Go to your [Agent Library](agent-library.md) and open the agent
2. Click **New Trigger** (this replaces the "New Task" button for trigger-based agents)
3. Give the trigger a **name** and **description**

#### Step 3: Copy the Webhook URL

Once the trigger is created, you'll see a status panel:

> **Trigger Status**
>
> Status: **Active**
>
> This trigger is ready to be used. Use the Webhook URL below to set up the trigger connection with the service of your choosing.
>
> **Webhook URL:** `https://backend.agpt.co/api/integrations/generic_webhook/webhooks/...`

Copy this webhook URL and provide it to the external platform or code that will be sending events to trigger your agent.

{% hint style="info" %}
Trigger-based agents cannot be started manually with the "New Task" button. The only way to execute them is by sending data to the webhook URL.
{% endhint %}

### Example Use Cases

- **GitHub webhook**: Trigger an agent whenever a pull request is opened
- **Payment processor**: Run an agent when a new payment is received
- **Form submission**: Process data when a user submits a form on your website
- **Custom integration**: Send data from any service that supports webhooks

## Schedule vs. Trigger

| | Schedule | Trigger |
|---|----------|---------|
| **How it starts** | Automatically at configured times | When an external event sends data to the webhook URL |
| **Input source** | Pre-configured when the schedule is created | Provided by the incoming webhook payload |
| **Use case** | Recurring tasks with fixed inputs (daily reports, weekly summaries) | Event-driven tasks (new PR opened, form submitted, payment received) |
| **Setup** | Through the "New Task" → "Schedule Task" flow | Through trigger blocks in the builder + "New Trigger" in the library |
