# Organize your agents, get notified, and enjoy a cleaner chat

*March 5 – March 12, 2026*

***

## Organize agents into folders

You can now **create folders in your library** to keep your agents organized. Drag and drop agents between folders, rename them, and color-code them to match your workflow. Whether you have five agents or fifty, everything stays tidy. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12290)

<figure><img src="../.gitbook/assets/folders-hero.png" alt="Organizing agents into color-coded folders in the AutoGPT library"><figcaption><p>Organizing agents into color-coded folders in the AutoGPT library</p></figcaption></figure>

## Get notified when background chats finish

AutoPilot now sends you a **notification when a background chat completes**. No more switching tabs to check — you'll see a toast alert the moment your agent finishes its work, so you can jump right back into the results. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12364)

<figure><img src="../.gitbook/assets/notifications-hero.png" alt="AutoPilot notification appearing when a background chat finishes its work"><figcaption><p>AutoPilot notification appearing when a background chat finishes its work</p></figcaption></figure>

## Cleaner reasoning & chat experience

We've redesigned how **reasoning steps and tool calls appear in chat**. Intermediate thinking is now collapsed by default, keeping your conversation clean and easy to read. Expand any step if you want the full details. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12282)

<figure><img src="../.gitbook/assets/reasoning-hero.png" alt="Collapsed reasoning steps keeping the AutoPilot conversation clean and focused"><figcaption><p>Collapsed reasoning steps keeping the AutoPilot conversation clean and focused</p></figcaption></figure>

## See outputs right after actions

Action results now **display inline immediately after each step completes**. You'll see files, data, and workspace outputs exactly where they belong — right alongside the action that created them. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12258)

<figure><img src="../.gitbook/assets/outputs-hero.png" alt="Action outputs displayed inline right after each step in AutoPilot"><figcaption><p>Action outputs displayed inline right after each step in AutoPilot</p></figcaption></figure>

***

<details>

<summary>Improvements</summary>

- **Credential selector redesign** [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12378) — Completely rebuilt the credential picker for a smoother, faster experience when connecting your accounts.

- **Pinned tool cards** [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12346) — Interactive tool cards now stay pinned outside collapsed reasoning, so results are always visible without expanding steps.

- **Workspace file uploads** [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12226) — Upload files directly to your workspace and reference them across agents and conversations.

- **Multimodal vision support** [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12273) — AutoPilot can now analyze images and PDFs you share in chat, powered by the latest vision models.

- **Per-turn summary stats** [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12257) — Each agent turn now shows a work-done summary so you can see exactly what happened at a glance.

- **Text-to-speech voice selection** [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12317) — Choose from multiple AI voices when listening to agent responses, with improved quality to avoid robotic-sounding output.

- **Batch undo in agent builder** [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12344) — Undo multiple changes at once when editing agents in the visual builder.

</details>

<details>

<summary>Fixes</summary>

- **Session message preservation** [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12302) — Title updates no longer overwrite session messages, keeping your full conversation intact.

- **Transcript reliability** [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12318) — Refactored transcripts to an atomic full-context model, preventing data loss during complex multi-step runs.

- **File download integrity** [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12349) — Workspace file downloads are now buffered to prevent truncation of large files.

- **Node layout handling** [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12354) — Improved how the builder handles discriminated unions and node positioning when editing agents.

- **Triggered agent runs** [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12298) — Manual run attempts for triggered agents are now handled gracefully instead of failing silently.

- **Password reset flow** [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12384) — Supabase error details now pass through correctly during password reset, giving clearer feedback when something goes wrong.

</details>

<details>

<summary>Under the hood</summary>

- **New models added** — Claude Sonnet 4.6, Grok 3, Mistral Large, Medium, Small & Codestral, Perplexity Sonar Reasoning Pro, Google Gemini 3.1 Flash & Pro, Phi-4, and Cohere Command A are now available.

- **File-ref protocol** [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12332) — Internal file references now use a consistent protocol, making cross-block file passing more reliable.

- **Langfuse tracing** [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12281) — Better observability for agent execution with enhanced trace correlation and feedback loops.

- **Sandbox auto-pause** [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12330) — Code execution sandboxes now automatically pause when idle, eliminating unnecessary billing.

- **Claude Code auth** [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12288) — Added subscription authentication support for Claude Code integration in SDK mode.

</details>
