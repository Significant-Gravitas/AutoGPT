# File uploads, cloud sandboxes, and browser automation

*February 26 – March 4, 2026*

***

## Upload files to AutoPilot

You can now attach files directly in the AutoPilot chat — documents, images, spreadsheets, audio, and video. Hit the **+** button, pick your files, and they appear as neat chips in the input bar with a spinner while they upload. Once sent, your attachments display inline in the conversation and the AI can read and reference them. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12220)

## Code runs in a cloud sandbox

AutoPilot now executes code in a secure cloud sandbox instead of locally. Files created during code execution, image generation, or document creation all live in the same shared environment — so a script can read a file that another tool just wrote, without anything getting lost between steps. The sandbox reconnects automatically if your session is interrupted. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12212)

{% hint style="info" %}
Tool outputs in chat now display richer detail — bash results show stdout and stderr, file edits show before-and-after diffs, and search results are formatted with icons by category.
{% endhint %}

## AutoPilot can browse the web for you

Two new browsing capabilities let AutoPilot interact with websites on your behalf. For quick lookups, it can fetch and extract content from any page in one shot. For multi-step tasks — like logging into a site, navigating through menus, and pulling data — it drives a full browser session that persists across steps within your conversation. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12230)

<figure><img src="../.gitbook/assets/web-browsing-hero.png" alt="AutoPilot browsing a website autonomously in a live browser session"><figcaption><p>AutoPilot navigating a website in a live browser session</p></figcaption></figure>

## Connect to any app — instantly

That integration you've been waiting for? You don't need to wait anymore. AutoPilot now supports **MCP** (Model Context Protocol) — an open standard backed by hundreds of ready-made connectors for apps like Notion, Slack, Jira, Stripe, Postgres, and more. Just tell AutoPilot what you want to connect to, and it finds and sets up the right connector automatically — nothing to install, nothing to configure. Search your Notion workspace, query a database, create a Jira ticket — all from the chat. If the service requires a login, you'll get the same familiar sign-in prompt you already know. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12213)

<figure><img src="../.gitbook/assets/mcp-notion-hero.png" alt="AutoPilot connecting to Notion via MCP and searching a workspace in real time"><figcaption><p>AutoPilot connecting to Notion and pulling release highlights — no integration setup required</p></figcaption></figure>

## Listen to responses and share them

A new speaker button on AutoPilot messages reads the response aloud using your browser's text-to-speech. There's also a share button that lets you send a response via your device's share menu, or copies it to clipboard as a fallback. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12256)

## See when the AI is summarizing

When a conversation gets long enough that AutoPilot needs to summarize earlier messages to stay within its memory limits, you'll now see a clear indicator — a spinner with "Summarizing earlier messages…" — instead of it happening silently in the background. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12250)

## Redesigned chat input

The chat input bar has been rebuilt with a cleaner layout — a spacious text area that grows smoothly as you type, with tool buttons and the send button in a tidy footer row. No more jarring height jumps when composing longer messages. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12207)

***

<details>

<summary>Improvements</summary>

* [More reliable streaming](https://github.com/Significant-Gravitas/AutoGPT/pull/12254) — AutoPilot now connects directly to the backend for real-time updates, avoiding an intermediate proxy that could cause timeouts on long-running tasks
* [Faster stall recovery](https://github.com/Significant-Gravitas/AutoGPT/pull/12254) — if the connection goes quiet, AutoPilot detects it in 10 seconds instead of 30 and reconnects automatically
* [Stop button works instantly](https://github.com/Significant-Gravitas/AutoGPT/pull/12254) — clicking Stop now immediately unlocks the input so you can keep working

</details>

<details>

<summary>Fixes</summary>

* [Node output no longer overflows](https://github.com/Significant-Gravitas/AutoGPT/pull/12222) — long text and JSON in the agent builder stays within its container, with output items capped for readability
* [Workspace file saves no longer conflict](https://github.com/Significant-Gravitas/AutoGPT/pull/12267) — when two operations write to the same file at the same time, you get a clear message instead of a raw error
* [Login no longer shows a blank page](https://github.com/Significant-Gravitas/AutoGPT/pull/12285) — signing in with email and password now loads the app immediately instead of requiring a manual refresh

</details>

<details>

<summary>Under the hood</summary>

* [Observability for the AI engine](https://github.com/Significant-Gravitas/AutoGPT/pull/12228) — every AutoPilot conversation turn is now traced with token counts, costs, and timing for faster debugging
* [Broadcast tracing enabled](https://github.com/Significant-Gravitas/AutoGPT/pull/12277) — monitoring coverage extended to the AI model routing layer
* [Legacy code removed](https://github.com/Significant-Gravitas/AutoGPT/pull/12276) — ~1,200 lines of old, unmaintained chat code cleaned out and replaced with a streamlined fallback path
* [Auto-synced developer tooling](https://github.com/Significant-Gravitas/AutoGPT/pull/12211) — switching branches now automatically keeps types and dependencies in sync

</details>
