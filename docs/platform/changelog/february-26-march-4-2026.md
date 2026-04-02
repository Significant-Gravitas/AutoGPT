# Connect any app, share files, and let AutoPilot browse for you

*February 26 – March 4, 2026*

***

## Connect to any app — instantly

That integration you've been waiting for? You don't need to wait anymore. AutoPilot now supports MCP (Model Context Protocol) — an open standard backed by hundreds of ready-made connectors for apps like Notion, Slack, Jira, Stripe, Postgres, and more. **Just tell AutoPilot what you want to connect to, and it finds and sets it up automatically** — nothing to install, nothing to configure. Search your Notion workspace, query a database, create a Jira ticket — all from the chat. If the service requires a login, you'll get the same familiar sign-in prompt you already know. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12213)

<figure><img src="../.gitbook/assets/mcp-notion-hero.png" alt="AutoPilot connecting to Notion via MCP and searching a workspace in real time"><figcaption><p>AutoPilot connecting to Notion and pulling release highlights — no integration setup required</p></figcaption></figure>

## Upload files to AutoPilot

You can now attach files directly in the AutoPilot chat — documents, images, spreadsheets, audio, and video. Hit the **+** button or drag and drop files into the chat. Once sent, your attachments display inline in the conversation and the AI can read and reference them. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12220)

<figure><img src="../.gitbook/assets/file-upload-hero.png" alt="Dragging a file into the AutoPilot chat"><figcaption><p>Drop files directly into the chat and AutoPilot picks them up instantly</p></figcaption></figure>

## AutoPilot can run code and create files for you

You can now drop a messy spreadsheet into AutoPilot and ask it to clean the data, find trends, and generate a polished chart. It can crunch numbers, build scripts, create documents, and produce downloadable files, all inside the conversation. Each step builds on the last, so you can go from raw data to finished deliverable just by asking. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12212)

<figure><img src="../.gitbook/assets/code-execution-hero.png" alt="AutoPilot analyzing business data and generating a sales performance dashboard"><figcaption><p>AutoPilot analyzing 14 months of business data and generating a dashboard — all from a single request</p></figcaption></figure>

## AutoPilot can browse the web for you

Two new browsing capabilities let AutoPilot interact with websites on your behalf. For quick lookups, it can fetch and extract content from any page in one shot. For multi-step tasks — like logging into a site, navigating through menus, and pulling data — it drives a full browser session that persists across steps within your conversation. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12230)

<figure><img src="../.gitbook/assets/web-browsing-hero.png" alt="AutoPilot browsing a website autonomously in a live browser session"><figcaption><p>AutoPilot navigating a website in a live browser session</p></figcaption></figure>

## Listen to responses or copy them instantly

Multitasking? Tap the speaker icon on any message to have AutoPilot read it aloud while you do something else. Need to save a response? Hit the copy button to grab it in one click — ready to paste into an email, doc, or chat. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12256)

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
