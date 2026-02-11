# January 29 – February 11, 2026

A round-up of everything we shipped over the last two weeks.

***

## Talk to AutoPilot with your voice

A mic button has been added to the chat input. Press it, speak your request, and your voice is transcribed instantly. The text box auto-focuses so you can hit Enter to send or tap the mic again to keep talking.

## Your files stick around

Files generated during conversations — images, documents, anything — are now saved to your personal workspace instead of disappearing when the session ends. You can browse and manage them directly through chat, and images show up inline in the conversation.

This also unblocked 20+ blocks (file storage, email attachments, screenshots, image generation, and more) that weren't working in chat.

## Upgraded to Claude Opus 4.6

AutoPilot now runs on Anthropic's newest and most capable model, with a larger context window and doubled output capacity. Extended thinking is enabled, so the AI reasons internally before responding — you get cleaner, more focused answers instead of raw chain-of-thought in the chat.

{% hint style="info" %}
Claude 3.7 Sonnet has been retired. Agents using it were auto-migrated to Claude 4.5 Sonnet.
{% endhint %}

## Customize marketplace agents

Ask AutoPilot to modify any marketplace agent before adding it to your library. "Customize that newsletter writer to post to Discord instead." AutoPilot adapts the workflow, asks follow-up questions if needed, and saves the customized version for you.

## Rebuilt chat streaming

The entire chat system has been rebuilt on the Vercel AI SDK. More reliable message delivery, better markdown formatting, tool outputs shown in clean collapsible panels, and errors surfaced as brief notifications instead of breaking the chat.

## Stay connected during long tasks

If your connection drops while AutoPilot is working on something (network hiccup, laptop sleep, switching tabs), it keeps going in the background, saves progress, and replays what you missed when you reconnect. No more starting over.

## Better agent creation

Progress bar during generation so you're not staring at a blank screen. A "Your agent is ready!" prompt with buttons to test immediately or provide your own inputs. Cleaner formatting when AutoPilot asks you clarifying questions. And helpful error messages instead of generic failures when something goes wrong.

## Redesigned homepage

The homepage now says "Tell me about your work — I'll find what to automate" instead of assuming you already know what to build. The quick-start buttons have been rewritten too:

| Before                          | After                                                  |
| ------------------------------- | ------------------------------------------------------ |
| Show me what I can automate     | I don't know where to start, just ask me stuff         |
| Design a custom workflow        | I do the same thing every week and it's killing me     |
| Help me with content creation   | Help me find where I'm wasting my time                 |

## New agents reuse your existing ones

When building a new agent, AutoPilot now searches your library for agents that can be incorporated as building blocks — two passes, first by overall goal, then by individual steps — so you don't rebuild from scratch.

***

<details>

<summary>Improvements</summary>

* Editing an agent now updates the original instead of creating a duplicate in your library
* "Tasks" tab renamed to "Agents" to better describe what's there
* Credit wallet no longer pops open automatically for new users or on balance changes
* Agent search results now suggest next steps, like offering to create a custom agent
* Chat no longer shows blocks that only work in the visual builder
* Linear search block now returns issue status, assignee, and project info with team filtering
* New Text Encoder block for escaping special characters in JSON payloads and config files

</details>

<details>

<summary>Fixes</summary>

* Improved credential matching so agents reliably use the correct API keys and permissions for each provider
* Agent inputs are now validated upfront with clear feedback on available fields
* Marketplace agents referenced as sub-agents in new builds no longer fail
* YouTube transcription block now cleanly reports either success or failure, not both
* Improved context management for longer chat sessions so the AI stays responsive
* Optimized the endpoint that loads your agent list
* Fixed an issue where hard refresh could briefly redirect to the wrong page

</details>
