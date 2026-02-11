# Voice input, persistent files, and a smarter AI

`January 29 – February 11, 2026`

A round-up of everything we shipped over the last two weeks.

***

## Talk to AutoPilot with your voice

A mic button has been added to the chat input. Press it, speak your request, and your voice is transcribed instantly. The text box auto-focuses so you can hit Enter to send or tap the mic again to keep talking.

_PRs: [#11871](https://github.com/Significant-Gravitas/AutoGPT/pull/11871), [#11893](https://github.com/Significant-Gravitas/AutoGPT/pull/11893)_

## Your files stick around

Files generated during conversations — images, documents, anything — are now saved to your personal workspace instead of disappearing when the session ends. You can browse and manage them directly through chat, and images show up inline in the conversation.

This also unblocked 20+ blocks (file storage, email attachments, screenshots, image generation, and more) that weren't working in chat.

_PR: [#11867](https://github.com/Significant-Gravitas/AutoGPT/pull/11867)_

## Upgraded to Claude Opus 4.6

AutoPilot now runs on Anthropic's newest and most capable model, with a larger context window and doubled output capacity. Extended thinking is enabled, so the AI reasons internally before responding — you get cleaner, more focused answers instead of raw chain-of-thought in the chat.

{% hint style="info" %}
Claude 3.7 Sonnet has been retired. Agents using it were auto-migrated to Claude 4.5 Sonnet.
{% endhint %}

_PRs: [#11983](https://github.com/Significant-Gravitas/AutoGPT/pull/11983), [#11841](https://github.com/Significant-Gravitas/AutoGPT/pull/11841), [#12052](https://github.com/Significant-Gravitas/AutoGPT/pull/12052)_

## Customize marketplace agents

Ask AutoPilot to modify any marketplace agent before adding it to your library. "Customize that newsletter writer to post to Discord instead." AutoPilot adapts the workflow, asks follow-up questions if needed, and saves the customized version for you.

_PR: [#11943](https://github.com/Significant-Gravitas/AutoGPT/pull/11943)_

## Rebuilt chat streaming

The entire chat system has been rebuilt on the Vercel AI SDK. More reliable message delivery, better markdown formatting, tool outputs shown in clean collapsible panels, and errors surfaced as brief notifications instead of breaking the chat.

_PRs: [#11901](https://github.com/Significant-Gravitas/AutoGPT/pull/11901), [#12063](https://github.com/Significant-Gravitas/AutoGPT/pull/12063)_

## Stay connected during long tasks

If your connection drops while AutoPilot is working on something (network hiccup, laptop sleep, switching tabs), it keeps going in the background, saves progress, and replays what you missed when you reconnect. No more starting over.

_PR: [#11877](https://github.com/Significant-Gravitas/AutoGPT/pull/11877)_

## Better agent creation

Progress bar during generation so you're not staring at a blank screen. A "Your agent is ready!" prompt with buttons to test immediately or provide your own inputs. Cleaner formatting when AutoPilot asks you clarifying questions. And helpful error messages instead of generic failures when something goes wrong.

_PRs: [#11974](https://github.com/Significant-Gravitas/AutoGPT/pull/11974), [#11975](https://github.com/Significant-Gravitas/AutoGPT/pull/11975), [#11985](https://github.com/Significant-Gravitas/AutoGPT/pull/11985), [#11884](https://github.com/Significant-Gravitas/AutoGPT/pull/11884), [#11993](https://github.com/Significant-Gravitas/AutoGPT/pull/11993)_

## Redesigned homepage

The homepage now says "Tell me about your work — I'll find what to automate" instead of assuming you already know what to build. The quick-start buttons have been rewritten too:

| Before                          | After                                                  |
| ------------------------------- | ------------------------------------------------------ |
| Show me what I can automate     | I don't know where to start, just ask me stuff         |
| Design a custom workflow        | I do the same thing every week and it's killing me     |
| Help me with content creation   | Help me find where I'm wasting my time                 |

_PR: [#11956](https://github.com/Significant-Gravitas/AutoGPT/pull/11956)_

## New agents reuse your existing ones

When building a new agent, AutoPilot now searches your library for agents that can be incorporated as building blocks — two passes, first by overall goal, then by individual steps — so you don't rebuild from scratch.

_PR: [#11889](https://github.com/Significant-Gravitas/AutoGPT/pull/11889)_

***

<details>

<summary>Improvements</summary>

* Editing an agent now updates the original instead of creating a duplicate in your library — [#11981](https://github.com/Significant-Gravitas/AutoGPT/pull/11981)
* "Tasks" tab renamed to "Agents" to better describe what's there — [#11982](https://github.com/Significant-Gravitas/AutoGPT/pull/11982)
* Credit wallet no longer pops open automatically for new users or on balance changes — [#11961](https://github.com/Significant-Gravitas/AutoGPT/pull/11961)
* Agent search results now suggest next steps, like offering to create a custom agent — [#11976](https://github.com/Significant-Gravitas/AutoGPT/pull/11976)
* Chat no longer shows blocks that only work in the visual builder — [#11892](https://github.com/Significant-Gravitas/AutoGPT/pull/11892)
* Linear search block now returns issue status, assignee, and project info with team filtering — [#11967](https://github.com/Significant-Gravitas/AutoGPT/pull/11967)
* New Text Encoder block for escaping special characters in JSON payloads and config files — [#11857](https://github.com/Significant-Gravitas/AutoGPT/pull/11857)

</details>

<details>

<summary>Fixes</summary>

* Improved credential matching so agents reliably use the correct API keys and permissions for each provider — [#11908](https://github.com/Significant-Gravitas/AutoGPT/pull/11908), [#11881](https://github.com/Significant-Gravitas/AutoGPT/pull/11881), [#11905](https://github.com/Significant-Gravitas/AutoGPT/pull/11905)
* Agent inputs are now validated upfront with clear feedback on available fields — [#11916](https://github.com/Significant-Gravitas/AutoGPT/pull/11916)
* Marketplace agents referenced as sub-agents in new builds no longer fail — [#11920](https://github.com/Significant-Gravitas/AutoGPT/pull/11920)
* YouTube transcription block now cleanly reports either success or failure, not both — [#11980](https://github.com/Significant-Gravitas/AutoGPT/pull/11980)
* Improved context management for longer chat sessions so the AI stays responsive — [#11937](https://github.com/Significant-Gravitas/AutoGPT/pull/11937)
* Optimized the endpoint that loads your agent list — [#12053](https://github.com/Significant-Gravitas/AutoGPT/pull/12053)
* Fixed an issue where hard refresh could briefly redirect to the wrong page — [#11894](https://github.com/Significant-Gravitas/AutoGPT/pull/11894), [#11900](https://github.com/Significant-Gravitas/AutoGPT/pull/11900), [#11903](https://github.com/Significant-Gravitas/AutoGPT/pull/11903), [#11904](https://github.com/Significant-Gravitas/AutoGPT/pull/11904)

</details>
