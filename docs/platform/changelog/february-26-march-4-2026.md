# File uploads, browser automation, and MCP tools in AutoPilot

*February 26 – March 4, 2026*

***

## Upload files directly into AutoPilot chat

You can now attach files — documents, images, spreadsheets, audio, and video — directly in the AutoPilot chat input. Hit the new **+** button, pick your files, and send. They appear as styled pills in your message, and AutoPilot can read and work with them immediately. Upload progress is shown in real time, and there are sensible per-file and per-account storage limits to keep things tidy. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12220)

## AutoPilot can browse the web for you

AutoPilot now has full multi-step browser automation. Ask it to visit a page, fill in a form, click buttons, take screenshots, or navigate a login flow — it handles the entire sequence inside a real browser session. Cookies and login state persist within your conversation, so you can chain steps together naturally: "Log in to my dashboard, then grab the latest report." [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12230)

## Connect any MCP tool server

AutoPilot can now discover and run tools from any [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server. Give it a server URL, and it will list the available tools, then execute them on your behalf — with the same one-click credential connection you already use elsewhere on the platform. Hundreds of integrations are available at [registry.modelcontextprotocol.io](https://registry.modelcontextprotocol.io/), including GitHub, Slack, Postgres, and more. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12213)

## Code execution got a major upgrade

AutoPilot now runs code in a cloud sandbox with full internet access and a persistent filesystem. Files you create with code, download, or generate all live in the same place and are immediately accessible to every tool. No more "file not found" errors when switching between writing code and reading files — everything shares one coherent workspace. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12212)

## The chat input has been redesigned

The AutoPilot text input has been completely rebuilt with a cleaner, more polished design. Multi-line messages now resize smoothly without any jarring jumps, and the overall layout feels more spacious and intentional. The stop button is now a more subtle dark color instead of a distracting red. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12207)

***

<details>

<summary>Improvements</summary>

* [Text-to-speech and share output](https://github.com/Significant-Gravitas/AutoGPT/pull/12256) — listen to AutoPilot's responses out loud or share them with others via new action buttons
* [Context compaction is now visible](https://github.com/Significant-Gravitas/AutoGPT/pull/12250) — when AutoPilot compresses its memory during long conversations, you'll see a visual indicator instead of it happening silently
* [Cleaner node output display](https://github.com/Significant-Gravitas/AutoGPT/pull/12222) — long outputs in the agent builder are now neatly truncated instead of overflowing their containers

</details>

<details>

<summary>Fixes</summary>

* [Streaming is dramatically more reliable](https://github.com/Significant-Gravitas/AutoGPT/pull/12254) — AutoPilot now connects directly to the backend for real-time responses, eliminating timeouts that previously cut off long conversations. Reconnection is automatic, and clicking Stop now immediately unlocks the input instead of requiring a page refresh
* [Workspace file conflicts resolved](https://github.com/Significant-Gravitas/AutoGPT/pull/12267) — fixed a rare issue where saving files at the same moment could cause an error
* [Login now refreshes the page properly](https://github.com/Significant-Gravitas/AutoGPT/pull/12285) — signing in with email and password now correctly loads your dashboard without needing a manual refresh

</details>

<details>

<summary>Under the hood</summary>

* [Legacy AutoPilot engine removed](https://github.com/Significant-Gravitas/AutoGPT/pull/12276) — cleaned out ~1,200 lines of old code and replaced it with a streamlined fallback engine, improving reliability and maintainability
* [Observability improved](https://github.com/Significant-Gravitas/AutoGPT/pull/12228) — added comprehensive usage tracking so the team can better monitor performance, identify issues faster, and optimize costs
* [OpenRouter broadcast enabled](https://github.com/Significant-Gravitas/AutoGPT/pull/12277) — expanded model routing capabilities for the AI engine
* [Developer workflow improved](https://github.com/Significant-Gravitas/AutoGPT/pull/12211) — types and dependencies now sync automatically when switching branches or committing code
* [Credit reset test hardened](https://github.com/Significant-Gravitas/AutoGPT/pull/12236) — fixed a flaky test related to billing cycle resets
* [Plans directory ignored in git](https://github.com/Significant-Gravitas/AutoGPT/pull/12229) — housekeeping for the developer environment

</details>
