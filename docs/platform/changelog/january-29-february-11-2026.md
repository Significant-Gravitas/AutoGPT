# Voice input, persistent files, and a smarter AI

*January 29 – February 11, 2026*

***

## Talk to AutoPilot with your voice

A mic button has been added to the chat input. Press it, speak your request, and your voice is transcribed instantly. The text box auto-focuses so you can hit Enter to send or tap the mic again to keep talking. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/11871)

## Your files stick around

Files generated during conversations — images, documents, anything — are now saved to your personal workspace instead of disappearing when the session ends. You can browse and manage them directly through chat, and images show up inline in the conversation. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/11867)

{% hint style="success" %}
This also unblocked **20+ blocks** (file storage, email attachments, screenshots, image generation, and more) that weren't working in chat.
{% endhint %}

## Upgraded to Claude Opus 4.6

AutoPilot now runs on Anthropic's newest and most capable model, with a larger context window and doubled output capacity. Extended thinking is enabled, so the AI reasons internally before responding — you get cleaner, more focused answers instead of raw chain-of-thought in the chat. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/11983)

{% hint style="info" %}
Claude 3.7 Sonnet has been retired. Agents using it were auto-migrated to Claude 4.5 Sonnet.
{% endhint %}

## Customize marketplace agents

Ask AutoPilot to modify any marketplace agent before adding it to your library. "Customize that newsletter writer to post to Discord instead." AutoPilot adapts the workflow, asks follow-up questions if needed, and saves the customized version for you. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/11943)

## Rebuilt chat streaming

The entire chat system has been rebuilt on the Vercel AI SDK. More reliable message delivery, better markdown formatting, tool outputs shown in clean collapsible panels, and errors surfaced as brief notifications instead of breaking the chat. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/11901)

## Stay connected during long tasks

If your connection drops while AutoPilot is working (network hiccup, laptop sleep, switching tabs), it keeps going in the background, saves progress, and replays what you missed when you reconnect. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/11877)

## New agents reuse your existing ones

When building a new agent, AutoPilot searches your library for agents that can be incorporated as building blocks — so you don't rebuild from scratch. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/11889)

## Better agent creation

Progress bar during generation so you're not staring at a blank screen. A "Your agent is ready!" prompt with buttons to test immediately or provide your own inputs. Cleaner formatting when AutoPilot asks you clarifying questions. And helpful error messages instead of generic failures. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/11974)

## Updated homepage text

The homepage now says "Tell me about your work — I'll find what to automate" instead of assuming you already know what to build. The quick-start buttons have been rewritten too: [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/11956)

{% tabs %}

{% tab title="After" %}
* "I don't know where to start, just ask me stuff"
* "I do the same thing every week and it's killing me"
* "Help me find where I'm wasting my time"
{% endtab %}

{% tab title="Before" %}
* ~~"Show me what I can automate"~~
* ~~"Design a custom workflow"~~
* ~~"Help me with content creation"~~
{% endtab %}

{% endtabs %}

***

<details>

<summary>Improvements</summary>

* [Editing agents updates the original](https://github.com/Significant-Gravitas/AutoGPT/pull/11981) instead of creating a duplicate in your library
* ["Tasks" tab renamed to "Agents"](https://github.com/Significant-Gravitas/AutoGPT/pull/11982) to better describe what's there
* [Wallet stays closed](https://github.com/Significant-Gravitas/AutoGPT/pull/11961) — no longer pops open automatically for new users or on balance changes
* [Search suggests next steps](https://github.com/Significant-Gravitas/AutoGPT/pull/11976), like offering to create a custom agent from your query
* [Smarter block filtering](https://github.com/Significant-Gravitas/AutoGPT/pull/11892) — chat no longer shows blocks that only work in the visual builder
* [Linear search block upgraded](https://github.com/Significant-Gravitas/AutoGPT/pull/11967) — now returns status, assignee, and project info with team filtering
* [New Text Encoder block](https://github.com/Significant-Gravitas/AutoGPT/pull/11857) for escaping special characters in JSON payloads and config files

</details>

<details>

<summary>Fixes</summary>

* [Improved credential matching](https://github.com/Significant-Gravitas/AutoGPT/pull/11908) so agents reliably use the correct API keys and permissions for each provider
* [Better input validation](https://github.com/Significant-Gravitas/AutoGPT/pull/11916) — agent inputs are now validated upfront with clear feedback on available fields
* [Marketplace agents work as sub-agents](https://github.com/Significant-Gravitas/AutoGPT/pull/11920) — referencing marketplace templates in new builds no longer fails
* [YouTube transcription fixed](https://github.com/Significant-Gravitas/AutoGPT/pull/11980) — cleanly reports either success or failure, not both
* [Long conversations stay responsive](https://github.com/Significant-Gravitas/AutoGPT/pull/11937) — improved context management for longer chat sessions
* [Agent list loads faster](https://github.com/Significant-Gravitas/AutoGPT/pull/12053) — optimized the endpoint that loads your agents
* [Login redirects fixed](https://github.com/Significant-Gravitas/AutoGPT/pull/11894) — resolved an issue where hard refresh could briefly show the wrong page

</details>
