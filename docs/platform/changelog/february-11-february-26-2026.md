# Telegram bots, agent folders, and a rebuilt builder

*February 11 – 26, 2026*

***

## AutoPilot got a major brain upgrade

AutoPilot is now powered by a completely new AI engine. It can hold real multi-step conversations, use tools behind the scenes, and handle complex requests that would have confused it before. When you need multiple things done at once, it runs them in parallel instead of one at a time — so everything finishes faster. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12103)

Ask it to run one of your agents and it will actually wait for the result and tell you what happened — no more "your agent started, go check later." [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12147)

{% hint style="info" %}
If your connection drops while AutoPilot is working on something, it keeps going in the background and picks up right where you left off when you reconnect. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12057)
{% endhint %}

## Automate your Telegram bots

Thirteen new Telegram blocks let you build agents that respond to messages, photos, voice notes, videos, and reactions. Set up triggers for incoming content, then reply with text, images, audio, documents, or video. Connect your bot token from [@BotFather](https://t.me/BotFather) and you're ready to go. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12141)

## The builder has been rebuilt

The old agent builder is gone. Everyone now uses the new flow editor — a cleaner, faster interface with improved node rendering and no more toggling between views. If you've been using the legacy builder, the experience should feel familiar but noticeably more polished. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12081)

## Organize agents into folders

Your agent library now supports folders. Create them with custom names, emoji icons, and colors, then drag-and-drop agents into them or right-click to move. Folders can be nested up to five levels deep, and a breadcrumb bar helps you navigate. Search still works across everything regardless of which folder you're in. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12101)

## Your files follow you everywhere

Files created during agent runs — whether from code execution, image generation, or document creation — now persist in your workspace automatically. They're accessible across conversations and sessions, and images render inline in chat. AutoPilot also retains full context about files and tools it used earlier in the conversation, so it won't "forget" what just happened. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12073) [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12164)

## Tell us what to build next

You can now submit feature requests directly through AutoPilot chat. Describe what you want — it checks for existing requests first and either creates a new one or adds your vote to a matching request so the team can see demand. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12102)

## Vague goals get refined, not rejected

If you ask AutoPilot to create an agent with a vague goal like "monitor social media," it now suggests a clearer, more actionable version instead of returning an error. Accept the suggestion with one click or refine it further. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12139)

## The stop button actually stops things

Clicking Stop now sends a real cancellation signal through the entire system. Previously it didn't always halt what was running behind the scenes — now it does. [*↗*](https://github.com/Significant-Gravitas/AutoGPT/pull/12171)

***

<details>

<summary>Improvements</summary>

* [Delete chat sessions](https://github.com/Significant-Gravitas/AutoGPT/pull/12112) — clean up old conversations with a trash icon and confirmation dialog
* [Five new list blocks](https://github.com/Significant-Gravitas/AutoGPT/pull/12105) — Flatten, Interleave, Zip, Difference, and Intersection for agents that work with lists
* [Web search enabled for agents](https://github.com/Significant-Gravitas/AutoGPT/pull/12108) — agents can now search the web as part of their workflows
* [PDF generation support](https://github.com/Significant-Gravitas/AutoGPT/pull/12216) — agents can create and manipulate PDF documents
* [Faster, smarter block search](https://github.com/Significant-Gravitas/AutoGPT/pull/11806) — the builder search now combines keyword and semantic matching so you find what you need even without the exact name
* [Parallel tool execution](https://github.com/Significant-Gravitas/AutoGPT/pull/12165) — when an agent needs multiple tools at once, they run simultaneously instead of one at a time
* [Signup answers carry forward](https://github.com/Significant-Gravitas/AutoGPT/pull/12119) — information from your signup form pre-populates AutoPilot setup so you don't repeat yourself
* [Improved agent creation UX](https://github.com/Significant-Gravitas/AutoGPT/pull/12117) — cleaner create and edit experience with better progress feedback
* [Create agents via API](https://github.com/Significant-Gravitas/AutoGPT/pull/12208) — developers can now programmatically create agents through the external API
* [Exact timestamps on hover](https://github.com/Significant-Gravitas/AutoGPT/pull/12087) — hover over "2 hours ago" to see the full date and time
* [Credentials and inputs always visible](https://github.com/Significant-Gravitas/AutoGPT/pull/12194) — required setup steps and login prompts no longer hide inside collapsible sections
* [Clarification and save cards always visible](https://github.com/Significant-Gravitas/AutoGPT/pull/12204) — questions from AutoPilot and save confirmations are shown upfront instead of inside accordions
* [Task lists expanded by default](https://github.com/Significant-Gravitas/AutoGPT/pull/12168) — AutoPilot's to-do and task lists start open so you see everything immediately
* [Cleaner builder nodes](https://github.com/Significant-Gravitas/AutoGPT/pull/12152) — the "Advanced" switch on nodes replaced with a simpler chevron toggle
* [Easier connection deletion](https://github.com/Significant-Gravitas/AutoGPT/pull/12083) — the delete button appears when hovering anywhere on the connection line, not just a tiny target
* [Workspace files render in markdown](https://github.com/Significant-Gravitas/AutoGPT/pull/12166) — images and links from your workspace display correctly in chat messages
* [Better password reset experience](https://github.com/Significant-Gravitas/AutoGPT/pull/12123) — expired or already-used reset links now explain what happened and how to get a new one
* [Snake minigame while you wait](https://github.com/Significant-Gravitas/AutoGPT/pull/12160) — play a quick game of Snake during longer operations
* [Cleaner chat interface](https://github.com/Significant-Gravitas/AutoGPT/pull/12094) — improved spacing and styling in AutoPilot conversations

</details>

<details>

<summary>Fixes</summary>

* [Streaming completely overhauled](https://github.com/Significant-Gravitas/AutoGPT/pull/12173) — tool outputs no longer get lost, and refreshing the page preserves your conversation
* [Sessions no longer get stuck](https://github.com/Significant-Gravitas/AutoGPT/pull/12191) — fixed cases where the AI would stop responding with no way to recover
* [Long-running tasks no longer time out](https://github.com/Significant-Gravitas/AutoGPT/pull/12175) — complex tasks can run as long as needed instead of being killed after five minutes
* [Background agents no longer stall or hallucinate](https://github.com/Significant-Gravitas/AutoGPT/pull/12167) — the AI won't claim to have completed something it didn't actually do
* [Error messages are now actually helpful](https://github.com/Significant-Gravitas/AutoGPT/pull/12205) — real error details instead of generic failures, and switching between chats resumes properly
* [Reconnection preserves your messages](https://github.com/Significant-Gravitas/AutoGPT/pull/12159) — dropping and reconnecting no longer clears your chat history
* [API errors stay out of the chat](https://github.com/Significant-Gravitas/AutoGPT/pull/12063) — backend errors no longer appear as garbled text in your conversation
* [Agent creation follow-ups work](https://github.com/Significant-Gravitas/AutoGPT/pull/12062) — asking follow-up questions after creating an agent no longer causes errors
* [API key expiration default removed](https://github.com/Significant-Gravitas/AutoGPT/pull/12092) — API keys no longer auto-expire the next day; you choose whether to set an expiration
* [Text selection works in the builder](https://github.com/Significant-Gravitas/AutoGPT/pull/11955) — selecting text in input fields no longer accidentally drags the entire node
* [Website extraction errors handled](https://github.com/Significant-Gravitas/AutoGPT/pull/12048) — clear error messages when a website can't be fetched instead of silent failures
* [Content no longer overflows cards](https://github.com/Significant-Gravitas/AutoGPT/pull/12060) — block output text stays within its container instead of getting cut off
* [Workspace file listing improved](https://github.com/Significant-Gravitas/AutoGPT/pull/12190) — files display with proper names, sizes, and types instead of raw data
* ["Show all my agents" works](https://github.com/Significant-Gravitas/AutoGPT/pull/12138) — asking AutoPilot to list all your agents now actually lists them instead of returning nothing
* [External link button visible](https://github.com/Significant-Gravitas/AutoGPT/pull/12209) — the "Open link" button in the safety modal is now properly styled and clickable
* [No more crashes on logout](https://github.com/Significant-Gravitas/AutoGPT/pull/12202) — the AutoPilot page no longer breaks if you log out while it's open
* [Code highlighting performance](https://github.com/Significant-Gravitas/AutoGPT/pull/12144) — code blocks load faster and use less memory
* [Agent generation UI refreshes](https://github.com/Significant-Gravitas/AutoGPT/pull/12070) — the interface now updates when agent creation finishes instead of appearing stuck
* [Concurrent saves no longer conflict](https://github.com/Significant-Gravitas/AutoGPT/pull/12177) — sending multiple rapid messages no longer causes database errors
* [Workspace file downloads work](https://github.com/Significant-Gravitas/AutoGPT/pull/12215) — files created by the AI now include proper download links
* [Chat message spacing fixed](https://github.com/Significant-Gravitas/AutoGPT/pull/12091) — messages no longer have awkward extra whitespace

</details>

<details>

<summary>Under the hood</summary>

* [Message queue infrastructure upgraded](https://github.com/Significant-Gravitas/AutoGPT/pull/12118) — moved from an end-of-life version to a current, supported release for better reliability
* [Production dependencies updated](https://github.com/Significant-Gravitas/AutoGPT/pull/12056) — security patches and stability improvements across the stack
* [Security scanning updated](https://github.com/Significant-Gravitas/AutoGPT/pull/12033) — upgraded the tools that automatically check for vulnerabilities
* [Block search responses optimized](https://github.com/Significant-Gravitas/AutoGPT/pull/12020) — reduced data sent during block lookups so searches feel snappier
* [Legacy builder code removed](https://github.com/Significant-Gravitas/AutoGPT/pull/12082) — with the new flow editor live, the old code has been fully cleaned out
* [Legacy agent views removed](https://github.com/Significant-Gravitas/AutoGPT/pull/12088) — retired old UI components for a lighter, faster app
* [Internal module structure improved](https://github.com/Significant-Gravitas/AutoGPT/pull/12068) — reduced circular dependencies for more reliable startup
* [UI components modernized](https://github.com/Significant-Gravitas/AutoGPT/pull/12136) — replaced legacy dialog components with the current design system
* [Error logging improved](https://github.com/Significant-Gravitas/AutoGPT/pull/11942) — more detailed API error tracking so issues get found and resolved faster
* [HumanInTheLoop block docs clarified](https://github.com/Significant-Gravitas/AutoGPT/pull/12069) — clearer output descriptions for the agent builder
* [Podman compatibility note added](https://github.com/Significant-Gravitas/AutoGPT/pull/12120) — documented a known limitation for users running Podman instead of Docker

</details>
