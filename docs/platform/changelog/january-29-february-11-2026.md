# Voice input, persistent files, and a smarter AI

*January 29 – February 11, 2026*

***

<table data-view="cards">

<thead>
<tr>
<th></th>
<th></th>
</tr>
</thead>

<tbody>

<tr>
<td><strong>Talk to AutoPilot with your voice</strong></td>
<td>A mic button has been added to the chat input. Press it, speak your request, and your voice is transcribed instantly. Hit Enter to send or tap the mic again to keep talking.</td>
</tr>

<tr>
<td><strong>Your files stick around</strong></td>
<td>Files generated during conversations are now saved to your personal workspace instead of disappearing when the session ends. Browse and manage them through chat, with images shown inline.</td>
</tr>

<tr>
<td><strong>Upgraded to Claude Opus 4.6</strong></td>
<td>AutoPilot now runs on Anthropic's newest model with a larger context window and doubled output capacity. Extended thinking means cleaner, more focused answers.</td>
</tr>

</tbody>
</table>

{% hint style="success" %}
The file workspace update also unblocked **20+ blocks** (file storage, email attachments, screenshots, image generation, and more) that weren't working in chat.
{% endhint %}

{% hint style="info" %}
Claude 3.7 Sonnet has been retired. Agents using it were auto-migrated to Claude 4.5 Sonnet.
{% endhint %}

***

## Customize marketplace agents

Ask AutoPilot to modify any marketplace agent before adding it to your library. "Customize that newsletter writer to post to Discord instead." AutoPilot adapts the workflow, asks follow-up questions if needed, and saves the customized version for you.

## Rebuilt chat streaming

The entire chat system has been rebuilt on the Vercel AI SDK. More reliable message delivery, better markdown formatting, tool outputs shown in clean collapsible panels, and errors surfaced as brief notifications instead of breaking the chat.

{% columns %}

{% column %}

### Stay connected during long tasks

If your connection drops while AutoPilot is working (network hiccup, laptop sleep, switching tabs), it keeps going in the background, saves progress, and replays what you missed when you reconnect.

{% endcolumn %}

{% column %}

### New agents reuse your existing ones

When building a new agent, AutoPilot searches your library for agents that can be incorporated as building blocks — so you don't rebuild from scratch.

{% endcolumn %}

{% endcolumns %}

## Better agent creation

Progress bar during generation so you're not staring at a blank screen. A "Your agent is ready!" prompt with buttons to test immediately or provide your own inputs. Cleaner formatting when AutoPilot asks you clarifying questions. And helpful error messages instead of generic failures.

## Redesigned homepage

The homepage now says "Tell me about your work — I'll find what to automate" instead of assuming you already know what to build. The quick-start buttons have been rewritten too:

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

<details>

<summary>Pull requests</summary>

**Voice input:** [#11871](https://github.com/Significant-Gravitas/AutoGPT/pull/11871), [#11893](https://github.com/Significant-Gravitas/AutoGPT/pull/11893) · **File workspace:** [#11867](https://github.com/Significant-Gravitas/AutoGPT/pull/11867) · **Claude Opus 4.6:** [#11983](https://github.com/Significant-Gravitas/AutoGPT/pull/11983), [#11841](https://github.com/Significant-Gravitas/AutoGPT/pull/11841), [#12052](https://github.com/Significant-Gravitas/AutoGPT/pull/12052) · **Marketplace customization:** [#11943](https://github.com/Significant-Gravitas/AutoGPT/pull/11943) · **Chat streaming:** [#11901](https://github.com/Significant-Gravitas/AutoGPT/pull/11901), [#12063](https://github.com/Significant-Gravitas/AutoGPT/pull/12063) · **Reconnection:** [#11877](https://github.com/Significant-Gravitas/AutoGPT/pull/11877) · **Agent creation:** [#11974](https://github.com/Significant-Gravitas/AutoGPT/pull/11974), [#11975](https://github.com/Significant-Gravitas/AutoGPT/pull/11975), [#11985](https://github.com/Significant-Gravitas/AutoGPT/pull/11985), [#11884](https://github.com/Significant-Gravitas/AutoGPT/pull/11884), [#11993](https://github.com/Significant-Gravitas/AutoGPT/pull/11993) · **Homepage:** [#11956](https://github.com/Significant-Gravitas/AutoGPT/pull/11956) · **Agent reuse:** [#11889](https://github.com/Significant-Gravitas/AutoGPT/pull/11889) · **Editing agents:** [#11981](https://github.com/Significant-Gravitas/AutoGPT/pull/11981) · **Tab rename:** [#11982](https://github.com/Significant-Gravitas/AutoGPT/pull/11982) · **Wallet:** [#11961](https://github.com/Significant-Gravitas/AutoGPT/pull/11961) · **Search suggestions:** [#11976](https://github.com/Significant-Gravitas/AutoGPT/pull/11976) · **Block filtering:** [#11892](https://github.com/Significant-Gravitas/AutoGPT/pull/11892) · **Linear block:** [#11967](https://github.com/Significant-Gravitas/AutoGPT/pull/11967) · **Text Encoder:** [#11857](https://github.com/Significant-Gravitas/AutoGPT/pull/11857) · **Credentials:** [#11908](https://github.com/Significant-Gravitas/AutoGPT/pull/11908), [#11881](https://github.com/Significant-Gravitas/AutoGPT/pull/11881), [#11905](https://github.com/Significant-Gravitas/AutoGPT/pull/11905) · **Input validation:** [#11916](https://github.com/Significant-Gravitas/AutoGPT/pull/11916) · **Sub-agents:** [#11920](https://github.com/Significant-Gravitas/AutoGPT/pull/11920) · **YouTube:** [#11980](https://github.com/Significant-Gravitas/AutoGPT/pull/11980) · **Context management:** [#11937](https://github.com/Significant-Gravitas/AutoGPT/pull/11937) · **Agent list:** [#12053](https://github.com/Significant-Gravitas/AutoGPT/pull/12053) · **Login redirects:** [#11894](https://github.com/Significant-Gravitas/AutoGPT/pull/11894), [#11900](https://github.com/Significant-Gravitas/AutoGPT/pull/11900), [#11903](https://github.com/Significant-Gravitas/AutoGPT/pull/11903), [#11904](https://github.com/Significant-Gravitas/AutoGPT/pull/11904)

</details>
