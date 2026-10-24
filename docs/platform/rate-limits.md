# Rate Limits

## Overview

The AutoGPT Platform limits how much any one caller can do in a given time, to control cost and prevent abuse. This page lists those limits: what each one covers, what it counts, how big it is, when it resets, and what you see when you reach it. If you self-host, it also names the setting that changes each one.

A limit is counted per **account** (everything you do while signed in, on any device), per **IP address**, or once for **everyone together**. [Client IP Addresses](client-ip.md) explains how the Platform determines the IP address it counts by.

{% hint style="info" %}
The numbers on this page are the Platform's built-in defaults. On AutoGPT Cloud, your plan sets the size of your chat allowance: see the [pricing page](https://agpt.co/pricing), and the **Usage limits** popover in the chat for your own figures.
{% endhint %}

### When you reach a limit

Most limits refuse the request with `429 Too Many Requests` and a message saying which limit you reached. Where the response carries a `Retry-After` header, it gives the number of seconds until you can try again. Some limits don't refuse anything: the work waits in a queue, or you get a simpler result. Each section below says which.

## Chat with Otto and your Experts

### Daily and weekly chat allowance

Every message you send to Otto or an Expert uses part of your chat allowance, measured by what the AI models cost to answer it. The allowance has two windows, and a message is refused when either is used up:

| Window | Resets                    | Default (self-hosted) |
| ------ | ------------------------- | --------------------- |
| Daily  | Every day at 00:00 UTC    | $2.50                 |
| Weekly | Every Monday at 00:00 UTC | $5.00                 |

- **Counted per:** account.
- **On AutoGPT Cloud**, each plan has its own allowance, which can change without a new release. The **Usage limits** popover in the chat shows how much of each window you have used and when it resets.
- **Free trial:** a trial has one allowance for the whole trial instead, to use at any pace. It doesn't reset, and it can't be reset with credits.
- **What counts:** chat messages, tasks you continue, replies read aloud in voice mode, and messages to Otto from chat apps you've linked, such as Telegram or Discord. Messages answered through your own connected ChatGPT account don't count.
- **What you see:** the chat says you've reached your daily or weekly usage limit and when it resets, and your message isn't sent. You can have a paused task resume, or a message send, when the allowance comes back. The API answers `429`, with the window and its reset time in the response body.
- **Reset with credits:** when the daily window is used up and the weekly one isn't, you can spend $5.00 of credits to reset the daily window, up to 5 times per UTC day. A sixth attempt is refused with `429`.

**Self-hosting:** set `CHAT_DAILY_COST_LIMIT_MICRODOLLARS` and `CHAT_WEEKLY_COST_LIMIT_MICRODOLLARS` on the backend, in microdollars (1 USD = `1000000`). The defaults are `2500000` and `5000000`. A negative value such as `-1` removes that window's limit, and `0` allows nothing. The [single-container image](single-container.md#autopilot-spend-caps) sets both to `-1`. Resets with credits need the credit system on (`ENABLE_CREDIT=true`). `CHAT_RATE_LIMIT_RESET_COST` sets the price of a reset in cents (`500`; `0` turns resets off), and `CHAT_MAX_DAILY_RESETS` the number per UTC day (`5`; `0` means no limit).

### Tasks running at once

| Limit                    | Default | Counted per | Setting                               |
| ------------------------ | ------- | ----------- | ------------------------------------- |
| Tasks running at once    | 5       | account     | `MAX_RUNNING_COPILOT_TURNS_PER_USER`  |
| Tasks running or waiting | 15      | account     | `MAX_INFLIGHT_COPILOT_TURNS_PER_USER` |

A task is one reply that Otto or an Expert is working on, in any chat. When 5 are running, a new message waits in a queue and starts by itself when one finishes; the chat says it has been queued. When 15 are running or waiting, a new message is refused with `429`, and the chat asks you to wait for one of your tasks to finish. Subtasks that Otto or an Expert starts count too, but they never take your last free running slot.

### Messages sent while Otto is working

While a reply is in progress you can keep sending follow-up messages, which Otto reads as it works.

- **Up to 30 a minute** per account, across all your chats. The minute starts with your first follow-up. Past that, the follow-up is still delivered, but the response is `429` so the app slows down.
- **Up to 10 waiting** per chat. A newer follow-up pushes out the oldest one still waiting.

Neither is configurable.

### Workflows Otto runs for you

In one chat, Otto can run each workflow up to 30 times and set up to 30 schedules for it. Past that, Otto tells you it has reached the maximum for this chat. Self-hosters can change both with `CHAT_MAX_AGENT_RUNS` and `CHAT_MAX_AGENT_SCHEDULES` (default `30` each).

## Experts

### Weekly budget per Expert

Each Expert you hire has a weekly budget for the work it runs on its own: its schedules, its triggers, and the workflows it runs for you. The default is $5.00 a week per Expert, and you can change it for each Expert or turn it off. The week starts on Monday at 00:00 UTC.

- At 80% of the budget, the Expert posts a warning in its chat.
- At the budget, the Expert pauses its schedules and says so in its chat. Its scheduled and triggered runs are skipped, and starting one of its runs by hand is refused with `409 Conflict`. The pause doesn't lift when the week ends: resume the Expert's schedules to continue, which also starts its count for the week over.

**Self-hosting:** `EXPERT_WEEKLY_CREDIT_BUDGET_DEFAULT` sets the default budget, in credits (100 credits = $1; default `500`; `0` turns it off).

### Approving further spending

Where spending approval is enabled, an Expert that has spent $2.50 in a week asks for your approval before it starts more work. Its new runs wait for review until you approve; one approval covers the rest of the week, and a declined run is stopped.

**Self-hosting:** `EXPERT_SPEND_APPROVAL_THRESHOLD_DEFAULT` sets the amount in credits (default `250`; `0` turns it off), and `EXPERT_SPEND_APPROVAL_WINDOW` counts it per `week` (the default) or per UTC `day`.

### Results posted in an Expert's chat

An Expert posts the result of each finished run in its chat, up to 10 a day per Expert (UTC). Later results that day appear only in your activity feed. Not configurable.

### Avatar generation

| Limit                          | Counted per | Resets                                    |
| ------------------------------ | ----------- | ----------------------------------------- |
| 10 generated avatars           | account     | 24 hours after the first one in the count |
| 15 seconds between generations | account     |                                           |

Failed generations count too. Past either limit, the request is refused with `429`, the message "Please wait before generating again. Limit: 10 avatars per day.", and a `Retry-After` header. Not configurable.

## Workflows

### Runs at once

You can have up to 25 runs of the same workflow running at once. More runs aren't refused: they wait in the queue as **Queued** and start as running ones finish.

- **Counted per:** account and workflow, whether a run was started by hand, by a schedule, by a trigger, by Otto or through the API.
- **Self-hosting:** `MAX_CONCURRENT_GRAPH_EXECUTIONS_PER_USER` (default `25`).

## Search

### Search across the Platform

The search box that searches everything in your account allows 120 searches a minute.

- **Counted per:** account. Each minute starts on the clock minute.
- **What you see:** `429`, with "Global search rate limit exceeded (120 requests per 60s). Try again shortly." There is no `Retry-After` header.
- Not configurable.

### Store search

| Searching  | Limit        | Counted per |
| ---------- | ------------ | ----------- |
| Signed in  | 120 a minute | account     |
| Signed out | 30 a minute  | IP address  |

- Each minute starts on the clock minute.
- **What you see:** `429`, with "Store search rate limit exceeded (120 requests per 60s). Try again shortly." (30 when signed out) and a `Retry-After` header.
- **Busy minutes:** store search matches on meaning as well as on words. In a minute when all searches together have used the Platform's budget for matching on meaning, new searches match on words alone until the next minute.

**Self-hosting:** `PUBLIC_SEARCH_USER_REQUESTS_PER_MINUTE` sets the limit per signed-in account (default `120`), `PUBLIC_SEARCH_REQUESTS_PER_MINUTE` the limit per IP address for signed-out searches (default `30`), and `PUBLIC_SEARCH_EMBEDDINGS_PER_MINUTE` the budget shared by all searches (default `1000`; `0` makes store search match on words alone). Signed-out searches are counted per visitor only when the frontend can tell visitors apart; see [Client IP Addresses](client-ip.md).

## Plans and billing

### Billing page and plan changes

Loading your plan, previewing a plan change, and starting, confirming, cancelling or resuming a trial or plan activation share one limit of 60 requests a minute.

- **Counted per:** account. Each minute starts on the clock minute.
- **What you see:** `429`, with "Subscription status rate limit exceeded (60 requests per 60s). Try again shortly." and a `Retry-After` header.
- Not configurable.

### Free trial places

AutoGPT Cloud may limit how many free trials run at the same time. When every place is taken, the trial isn't offered, and starting one is refused with `409 Conflict` and "The trial is full right now. Please check back later."

## Onboarding

### Personalizing your account

When you tell the Platform about yourself during onboarding, by voice or typing, it uses AI to personalize your account. On AutoGPT Cloud, an account without a plan can do this 10 times an hour and 30 times a day.

- **Counted per:** account. The hour starts on the clock hour, and the day at 00:00 UTC. Submitting the same profile again doesn't count; a skipped step doesn't count.
- **What you see:** `429`, with "You have made several personalization attempts. Your progress is saved; please try again later." and a `Retry-After` header.
- Not configurable, and off on a self-hosted install.

## Connecting integrations

### Sign-in with a code

Some integrations connect by showing a code to enter on the provider's site.

- **Starting:** once every 3 seconds per account and integration. Sooner is refused with `429` and "A device authorization was just started. Please wait a moment."
- **Checking for approval:** at most as often as the provider asks, every 5 seconds unless it says otherwise. A sooner check isn't refused: the answer is `slow_down`, as the provider would give.

Neither is configurable.

## Signing in and account emails

### Account emails

When the Platform requires new accounts to verify their email address:

| Limit                                                                                               | Counted per   | What you see                                                      |
| --------------------------------------------------------------------------------------------------- | ------------- | ----------------------------------------------------------------- |
| 5 sign-ups and **Resend email** clicks together, per 10 minutes                                     | IP address    | `429` and "Too many attempts. Please try again in a few minutes." |
| 1 verification email per 10 minutes                                                                 | email address | No new email is sent; use the one you already have                |
| 1 set-password email per 10 minutes, when you sign up again with an address that isn't verified yet | email address | No new email is sent; use the one you already have                |

The 10-minute windows for the first row start on the clock: at :00, :10, :20 and so on. **Resend email** also waits 60 seconds between clicks. None of these are configurable. A self-hosted install requires verification only with `AUTH_REQUIRE_EMAIL_VERIFICATION=true` on the frontend.

### The authentication API

Requests to the authentication API under `/api/auth/` are limited per IP address and path:

| Paths                                                           | Limit              |
| --------------------------------------------------------------- | ------------------ |
| `/sign-in/*`, `/sign-up/*`, `/change-password`, `/change-email` | 3 per 10 seconds   |
| `/request-password-reset`, `/send-verification-email`           | 3 per 60 seconds   |
| Every other path                                                | 100 per 10 seconds |

Each path counts on its own. A path allows that many requests in a row, and starts counting again once a whole window has passed since the last request it let through. Past the limit, the answer is `429` with `{"message":"Too many requests. Please try again later."}` and an `X-Retry-After` header in seconds.

These limits apply to production builds of the frontend and aren't configurable. On a self-hosted install behind a reverse proxy, see [Client IP Addresses](client-ip.md).

## Not on this page

- Limits of other services that blocks and Otto use, such as an AI provider's own rate limits. Their errors are passed on as they come.
- Size limits, such as the largest file you can upload or the storage that comes with your plan.
