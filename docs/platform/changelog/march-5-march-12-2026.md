# Organize your agents, get notified, and enjoy a cleaner chat

*March 5 – March 12, 2026*

***

## Organize agents into folders

You can now **create folders in your library** to keep your agents organized. Drag and drop agents between folders, rename them, and color-code them to match your workflow. Whether you have five agents or fifty, everything stays tidy. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12290)

<figure><img src="../.gitbook/assets/folders-hero.png" alt="Organizing agents into color-coded folders in the AutoGPT library"><figcaption><p>Organizing agents into color-coded folders in the AutoGPT library</p></figcaption></figure>

## Get notified when background chats finish

AutoPilot now sends you a **notification when a background chat completes**. No more switching tabs to check — you'll see a toast alert the moment your agent finishes its work, so you can jump right back into the results. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12258)

<figure><img src="../.gitbook/assets/notifications-hero.png" alt="AutoPilot notification appearing when a background chat finishes its work"><figcaption><p>AutoPilot notification appearing when a background chat finishes its work</p></figcaption></figure>

## Cleaner reasoning & chat experience

We've redesigned how **reasoning steps and tool calls appear in chat**. Intermediate thinking is now collapsed by default, keeping your conversation clean and easy to read. Expand any step if you want the full details. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12346)

<figure><img src="../.gitbook/assets/reasoning-hero.png" alt="Collapsed reasoning steps keeping the AutoPilot conversation clean and focused"><figcaption><p>Collapsed reasoning steps keeping the AutoPilot conversation clean and focused</p></figcaption></figure>

## See outputs right after actions

Action results now **display inline immediately after each step completes**. You'll see files, data, and workspace outputs exactly where they belong — right alongside the action that created them. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12258)

<figure><img src="../.gitbook/assets/outputs-hero.png" alt="Action outputs displayed inline right after each step in AutoPilot"><figcaption><p>Action outputs displayed inline right after each step in AutoPilot</p></figcaption></figure>

***

<details>

<summary>Improvements</summary>

- **Credential selector redesign** [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12378) — Completely rebuilt the credential picker for a smoother, faster experience when connecting your accounts.

- **Pinned tool cards** [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12346) — Frequently used tools are now pinned at the top of the tool panel for quick access.

- **Workspace file uploads** — Upload files directly to your workspace and reference them across agents and conversations.

- **Multimodal vision support** — AutoPilot can now analyze images you share in chat, powered by the latest vision models.

- **Summary stats on dashboard** — View at-a-glance execution stats and usage trends from your home dashboard.

- **Text-to-speech voice selection** — Choose from multiple AI voices when listening to agent responses.

- **Batch undo in agent builder** — Undo multiple changes at once when editing agents in the visual builder.

</details>

<details>

<summary>Fixes</summary>

- **Agent execution reliability** — Fixed several edge cases where agents would stall mid-execution or fail to report completion.

- **Chat message ordering** — Messages now always appear in the correct chronological order, even during fast exchanges.

- **Credential refresh** — OAuth tokens now refresh correctly in the background, preventing unexpected disconnections.

- **Builder node connections** — Fixed issues where links between blocks would occasionally break when moving nodes.

- **Webhook triggers** — Resolved cases where certain webhook payloads would fail to start agent runs.

- **Mobile layout** — Improved touch targets and scroll behavior on smaller screens.

</details>

<details>

<summary>Under the hood</summary>

- **New models added** — Claude Sonnet 4.6, Grok 3, Mistral Small & Saba, Perplexity Sonar Pro, Google Gemini 3.1 Flash & Pro, Phi-4, and Cohere Command A are now available.

- **File-ref protocol** — Internal file references now use a consistent protocol, making cross-block file passing more reliable.

- **Langfuse tracing improvements** — Better observability for agent execution with enhanced trace correlation and feedback loops.

- **Sandbox auto-pause** — Code execution sandboxes now automatically pause when idle, reducing resource consumption.

- **Claude Code auth** — Added authentication support for Claude Code integration.

</details>
