# A brain dump that helps you answer, a calmer trial welcome, and a guided first run

*October 27, 2026*

**Platform versions:** `v0.10.0`

## Tell Otto about your work, with a little help

The onboarding step where you tell Otto about your work now helps you answer. It says why it matters (Otto uses it to pick your Experts and plan your first week), and shows questions Otto might ask, picked for the role you chose a step earlier, with **Another question** when one doesn't fit. Example answers float around the page on wide screens. If your browser blocks the microphone, or there isn't one, the step tells you, switches you to typing and gives you a way to **try the microphone again**. Skipping still works, and going **Back** to write something after you skipped now takes your answer and builds your team from it. [#16252](https://github.com/Significant-Gravitas/AutoGPT/pull/16252) [#16067](https://github.com/Significant-Gravitas/AutoGPT/pull/16067) [#16009](https://github.com/Significant-Gravitas/AutoGPT/pull/16009) [#16207](https://github.com/Significant-Gravitas/AutoGPT/pull/16207)

<figure><img src="../.gitbook/assets/v0100-brain-dump-hero.gif" alt="The brain dump step at rest with Talk and Write, then typing with an Otto might ask question, then recording"><figcaption><p>Questions Otto might ask, picked for your role</p></figcaption></figure>

## A calmer welcome before your trial

The payment step near the end of onboarding is now one welcome card instead of a grid of plans. It names the Experts you just hired and lays out your trial on a short timeline: when it starts, when your first payment is due and the last day you can cancel without paying. [#16036](https://github.com/Significant-Gravitas/AutoGPT/pull/16036)

## A guided first run over the real app

New accounts get a short walkthrough of the app itself: it dims the page and lights up one real control at a time, Chats, Team, Files and your allowance in the wallet, and ends on a suggested first task. A few one-time tips appear later, the first time they matter, such as when you open a second chat. **Show me around** in the profile menu replays it any time. The sidebar now starts expanded, so your chats are in view from the first minute, and the wallet stays in the sidebar when you collapse it. [#16107](https://github.com/Significant-Gravitas/AutoGPT/pull/16107) [#16024](https://github.com/Significant-Gravitas/AutoGPT/pull/16024)

## Hire an Expert and hand over the task in one click

When Otto suggests hiring an Expert for something you asked for, the card now has one button, **Hire and start**. It hires the Expert and starts the task in their own chat, so you no longer have to say "go ahead" twice. The card then shows where the task stands, and if it couldn't start, it says why. [#16101](https://github.com/Significant-Gravitas/AutoGPT/pull/16101)

## See what pays for what

A new **How usage and credits work** dialog answers four questions in plain words: what your chat usage covers, what automation credits are for, when your card is charged and how to get more. Open it from **How it works** in the wallet or the info icon on Billing & usage. The wallet itself is shorter, and a trial shows one line with the day your plan starts. [#16095](https://github.com/Significant-Gravitas/AutoGPT/pull/16095)

## Pick an Airtable base from a list

Otto no longer asks you to copy an ID out of a URL. After you connect Airtable, the setup card shows **Choose a base** with your bases by name, greys out the ones that step can't use and adds a search box when you have many. [#16094](https://github.com/Significant-Gravitas/AutoGPT/pull/16094)

<details>

<summary>✨ Improvements</summary>

- **Onboarding and Otto's first greeting look sharper:** larger text, clear marks while your team is prepared, Otto's picture in the greeting and your first tasks as a grid of cards with the logos of the tools they use. ([#16236](https://github.com/Significant-Gravitas/AutoGPT/pull/16236))
- **Otto's first greeting names every problem you described,** including the ones Otto will work through with you, and each gets a suggested first task. ([#16165](https://github.com/Significant-Gravitas/AutoGPT/pull/16165))
- **Settings → Memory shows what Otto knows about your work** from onboarding, with a **Forget this** button. ([#16014](https://github.com/Significant-Gravitas/AutoGPT/pull/16014))
- **Files and Recent work show what you asked for:** Otto's working files and your Experts' skill files no longer clutter them or count toward your storage. ([#16028](https://github.com/Significant-Gravitas/AutoGPT/pull/16028), [#16039](https://github.com/Significant-Gravitas/AutoGPT/pull/16039))
- **Every step in a chat has a readable label,** and a lookup that finds nothing says so instead of showing an empty `[]`. ([#16034](https://github.com/Significant-Gravitas/AutoGPT/pull/16034))
- **A file link in a reply opens the file in the side panel** instead of downloading it. ([#16065](https://github.com/Significant-Gravitas/AutoGPT/pull/16065))
- **The paywall tells you when your scheduled workflows are paused** until you pick a plan. ([#16072](https://github.com/Significant-Gravitas/AutoGPT/pull/16072))
- **A new thinking indicator:** a single line that draws itself while Otto works. ([#15941](https://github.com/Significant-Gravitas/AutoGPT/pull/15941))
- **New models:** Kimi K3.1 Pro and GPT-6.1 Terra. ([#16096](https://github.com/Significant-Gravitas/AutoGPT/pull/16096), [#16261](https://github.com/Significant-Gravitas/AutoGPT/pull/16261))
- **Cloudflare is in the list of tools you can connect.** ([#16217](https://github.com/Significant-Gravitas/AutoGPT/pull/16217))

</details>

<details>

<summary>🎨 UI/UX Improvements</summary>

- **After onboarding, the Home chat box asks what Otto should take on first.** ([#16209](https://github.com/Significant-Gravitas/AutoGPT/pull/16209))
- **The What's new card skips releases from before you joined.** ([#16160](https://github.com/Significant-Gravitas/AutoGPT/pull/16160))
- **"Thought for" shows the same time before and after a reload.** ([#16073](https://github.com/Significant-Gravitas/AutoGPT/pull/16073))
- **Home's team tile says "Open team"** when it already lists everyone. ([#16068](https://github.com/Significant-Gravitas/AutoGPT/pull/16068))
- **The storage label on Files is easier to read.** ([#16102](https://github.com/Significant-Gravitas/AutoGPT/pull/16102))
- **A new Expert's first questions keep the keyboard,** so letter shortcuts pick options instead of typing into the chat. ([#16019](https://github.com/Significant-Gravitas/AutoGPT/pull/16019))
- **Screen readers announce what the chat's side panels hold on phones.** ([#16213](https://github.com/Significant-Gravitas/AutoGPT/pull/16213))

</details>

<details>

<summary>🐛 Bug Fixes</summary>

- **Question cards from Otto stay on screen** once they are asked. ([#16082](https://github.com/Significant-Gravitas/AutoGPT/pull/16082))
- **An Expert's setup questions are answered in the chat that asked them,** even when you open that Expert from the Team page. ([#16018](https://github.com/Significant-Gravitas/AutoGPT/pull/16018))
- **The hire step recommends a team from your brain dump again,** with three Experts when it has to fall back. ([#16010](https://github.com/Significant-Gravitas/AutoGPT/pull/16010))
- **Signing up with an address that can't receive mail says so** instead of telling you to check your inbox. ([#16057](https://github.com/Significant-Gravitas/AutoGPT/pull/16057))
- **An email sign-up is greeted as "there"** instead of by the start of the address. ([#16208](https://github.com/Significant-Gravitas/AutoGPT/pull/16208))
- **A message that mentions one connected account** no longer ends in a stray "__". ([#16142](https://github.com/Significant-Gravitas/AutoGPT/pull/16142))
- **Turning off browser notifications stops them.** ([#16171](https://github.com/Significant-Gravitas/AutoGPT/pull/16171))
- **Home's credits warning counts only workflow schedules,** and older scheduled runs show as "Scheduled run" with the workflow's name. ([#16063](https://github.com/Significant-Gravitas/AutoGPT/pull/16063), [#16058](https://github.com/Significant-Gravitas/AutoGPT/pull/16058))
- **An AI step the provider refuses says so in plain words** instead of an unknown error. ([#16059](https://github.com/Significant-Gravitas/AutoGPT/pull/16059))

</details>
