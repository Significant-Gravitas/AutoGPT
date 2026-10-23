# Chats that pick up where they paused, your whole team in one click, and redesigned Expert pages

*October 22, 2026*

**Platform version:** `v0.9.0`

## Pick up where Otto paused

Reaching a usage limit no longer has to end a task with a red error. Otto can save a checkpoint of what is done and what comes next, then show a **Paused** card. A long run offers **Keep going** or **Stop here**. When a daily or weekly allowance runs out, the card shows when it resets and offers to resume after that. If Otto was stopped before saving a checkpoint, the card says so. [#15388](https://github.com/Significant-Gravitas/AutoGPT/pull/15388) [#15401](https://github.com/Significant-Gravitas/AutoGPT/pull/15401)

You can also write your next message while you are out of allowance and choose **Send at reset**. A chip shows its scheduled time, with **Edit** and **Cancel** while it is waiting. The draft and pending send survive a reload, and a failed scheduling request puts the message back where you can recover it. This is available when your allowance has a reset time; an exhausted trial needs a plan instead. [#15390](https://github.com/Significant-Gravitas/AutoGPT/pull/15390) [#15403](https://github.com/Significant-Gravitas/AutoGPT/pull/15403) [#15591](https://github.com/Significant-Gravitas/AutoGPT/pull/15591) [#15592](https://github.com/Significant-Gravitas/AutoGPT/pull/15592)

## Know what is left, and what an upgrade costs

Usage notices now explain which allowance ran out and when you can continue. A running-low hint helps you decide whether to start a longer task. New free trials receive one chat allowance for the whole trial, which you can use at your own pace; existing trials keep their saved terms. [#15403](https://github.com/Significant-Gravitas/AutoGPT/pull/15403) [#15184](https://github.com/Significant-Gravitas/AutoGPT/pull/15184)

Starting Pro during an active trial shows the charge and next payment date before you confirm, and explains that starting now ends the trial early. Pro-to-Max upgrades show a proration quote first. A trial's first paid conversion to Pro or Max starts with fresh daily and weekly usage once payment is confirmed; ordinary paid-plan upgrades do not reset what you have already used. [#15215](https://github.com/Significant-Gravitas/AutoGPT/pull/15215) [#15208](https://github.com/Significant-Gravitas/AutoGPT/pull/15208)

## Routines tell you when they need help

A scheduled chat that cannot run now leaves a note explaining why, with an action to help you fix it. A one-time task blocked by a daily or weekly allowance can wait for the reset instead of disappearing. [#15374](https://github.com/Significant-Gravitas/AutoGPT/pull/15374)

If three runs in a row fail outright for the same reason, the schedule pauses and appears under **Needs you** on Home. Open its chat to reconnect the missing service or resume it. Reconnecting a provider from another page also resumes chats paused on that connection. If a single step keeps failing but the rest of the run still delivers, you get a note and the schedule keeps running. [#15375](https://github.com/Significant-Gravitas/AutoGPT/pull/15375) [#15377](https://github.com/Significant-Gravitas/AutoGPT/pull/15377)

For recurring chats that keep using up your allowance, **Run less often** offers a slower schedule directly from the usage-limit note. You can keep the routine useful without increasing your plan. [#15400](https://github.com/Significant-Gravitas/AutoGPT/pull/15400)

## Start with your whole team

After you tell Otto about your work, the hire step offers one button that hires every Expert Otto recommends. Untick a card first to leave someone out, or choose **I'll add members later**. [#15836](https://github.com/Significant-Gravitas/AutoGPT/pull/15836)

Otto's suggested first tasks now stay on Home under the composer until you hide each one. A task you started shows as in progress, so one click no longer clears the rest of your plan. [#15837](https://github.com/Significant-Gravitas/AutoGPT/pull/15837)

A new Expert's intake card takes several answers where a question allows it, lets you review your answers before sending, works from the keyboard anywhere in the card, and has controls sized for touch. Once answered, setup questions and Otto's multiple-choice questions stay in the chat as a readable list of questions and answers, including after a reload and in shared chats. [#15874](https://github.com/Significant-Gravitas/AutoGPT/pull/15874) [#15873](https://github.com/Significant-Gravitas/AutoGPT/pull/15873)

The login and signup pages now match the rest of the app and load without their heavy animations. [#15876](https://github.com/Significant-Gravitas/AutoGPT/pull/15876)

## See what each Expert has been doing

Expert pages have a new **Overview** with conversation and workflow-run counts, recent activity, and the latest outputs. **Soul** has its own tab for identity, voice and boundaries, and **Files** brings the Expert's files together with search and filters. The Team page has clearer search, category filters and empty states, and the layouts fit smaller screens. [#15500](https://github.com/Significant-Gravitas/AutoGPT/pull/15500)

<figure><img src="../.gitbook/assets/v090-expert-overview-hero.png" alt="An Expert's redesigned Overview with its budget, tabs, conversation count, workflow runs and outputs"><figcaption><p>Your Expert's activity and settings, together on its page</p></figcaption></figure>

## Ask Otto to email you

Ask Otto to "email me the result" or to send you a weekly digest, and the message arrives at your account's verified address from AutoGPT's own mail, with no Gmail or other sign-in first. Workflows can do the same with the new **Send Email To Me** block, whose body can be Markdown, plain text or HTML, with images. [#15838](https://github.com/Significant-Gravitas/AutoGPT/pull/15838)

## Chat more comfortably on your phone

The composer gives your message more room and keeps its controls above the phone's safe area. On touch devices, **Take photo** in the **+** menu sends a camera shot through the usual attachment flow. Extra controls live in **More options**. Attached and sent images now show as thumbnails, in the composer and in the chat, instead of plain file cards. [#15545](https://github.com/Significant-Gravitas/AutoGPT/pull/15545) [#15882](https://github.com/Significant-Gravitas/AutoGPT/pull/15882)

The keyboard stays closed until you need it and dismisses after sending. Message actions are visible without hovering, long links wrap, and the sidebar and the navigation menu close when you pick a chat or a page. A quick second tap on **Send** no longer stops the message you just sent. Your sidebar's open or closed choice also survives a reload. [#15769](https://github.com/Significant-Gravitas/AutoGPT/pull/15769) [#15692](https://github.com/Significant-Gravitas/AutoGPT/pull/15692) [#15547](https://github.com/Significant-Gravitas/AutoGPT/pull/15547) [#15574](https://github.com/Significant-Gravitas/AutoGPT/pull/15574)

## Self-host with the published image, including on Windows

The self-hosting guides now start with the published Docker Hub image, and the single-container install is out of experimental status. Windows has a PowerShell installer alongside the Linux and macOS installer, with the same install, update and lifecycle options. See [Setting Up AutoGPT](../getting-started.md) and the [installer guide](../installer.md). [#14553](https://github.com/Significant-Gravitas/AutoGPT/pull/14553)

Experts, Files, Skills, memory and the onboarding wizard no longer depend on a feature-flag service to switch on. [#15799](https://github.com/Significant-Gravitas/AutoGPT/pull/15799)

A few defaults change for self-hosted installs:

- **Sign-ups from disposable email addresses are refused** unless you set `AUTH_ALLOW_DISPOSABLE_EMAILS=true`. ([#15824](https://github.com/Significant-Gravitas/AutoGPT/pull/15824))
- **The backend's internal services require a shared key on every call,** and Docker Compose no longer publishes their ports on the host. Setup generates `INTERNAL_API_KEY`; an existing install derives it from its encryption key, with nothing to configure. ([#15949](https://github.com/Significant-Gravitas/AutoGPT/pull/15949))
- **Prometheus metrics move to internal ports** (8013 for the REST API, 8014 for the WebSocket server) when `ENABLE_METRICS` is on. ([#15966](https://github.com/Significant-Gravitas/AutoGPT/pull/15966))
- **Google Analytics loads only when you set a measurement ID.** ([#15748](https://github.com/Significant-Gravitas/AutoGPT/pull/15748))
- **The Docker images run Node 24.22.1,** which refuses certificates from the recently distrusted eMudhra intermediate authorities. ([#15888](https://github.com/Significant-Gravitas/AutoGPT/pull/15888))

<details>

<summary>✨ Improvements</summary>

- **What's new in the app:** a short notification shows each new release's highlights, with a link to these notes. ([#12652](https://github.com/Significant-Gravitas/AutoGPT/pull/12652))
- **Long chats feel lighter:** opening and scrolling a long conversation renders the visible messages, while the composer stays steady as a reply streams. ([#15740](https://github.com/Significant-Gravitas/AutoGPT/pull/15740), [#15716](https://github.com/Significant-Gravitas/AutoGPT/pull/15716), [#15710](https://github.com/Significant-Gravitas/AutoGPT/pull/15710), [#15688](https://github.com/Significant-Gravitas/AutoGPT/pull/15688))
- **The Builder and workflow pages respond faster** while you drag, type, search blocks or watch a run, and chat pages load diagrams and code highlighting only when a reply needs them. ([#15726](https://github.com/Significant-Gravitas/AutoGPT/pull/15726), [#15711](https://github.com/Significant-Gravitas/AutoGPT/pull/15711), [#15689](https://github.com/Significant-Gravitas/AutoGPT/pull/15689), [#15683](https://github.com/Significant-Gravitas/AutoGPT/pull/15683), [#15713](https://github.com/Significant-Gravitas/AutoGPT/pull/15713), [#15681](https://github.com/Significant-Gravitas/AutoGPT/pull/15681), [#15712](https://github.com/Significant-Gravitas/AutoGPT/pull/15712))
- **Otto can look at images in your files,** including a PDF page it rendered or a frame it pulled from a video. ([#15725](https://github.com/Significant-Gravitas/AutoGPT/pull/15725))
- **Memory is more careful:** follow-up replies get relevant saved memories again, concrete details from your requests are kept, a fact you asked Otto to forget stays forgotten, and something Otto only inferred is marked as unconfirmed. ([#15003](https://github.com/Significant-Gravitas/AutoGPT/pull/15003), [#15677](https://github.com/Significant-Gravitas/AutoGPT/pull/15677), [#14972](https://github.com/Significant-Gravitas/AutoGPT/pull/14972), [#15798](https://github.com/Significant-Gravitas/AutoGPT/pull/15798))
- **Otto prefers the services you have connected,** such as a linked Slack or Linear, over ones you would still have to set up. ([#15011](https://github.com/Significant-Gravitas/AutoGPT/pull/15011), [#15499](https://github.com/Significant-Gravitas/AutoGPT/pull/15499))
- **Morning briefings start in a fresh Otto chat,** so each day's briefing has its own place. ([#15222](https://github.com/Significant-Gravitas/AutoGPT/pull/15222))
- **Connections renew more reliably:** eligible accounts refresh before expiry, idle connections get maintenance, and rotating refresh tokens are protected from competing refreshes. ([#15619](https://github.com/Significant-Gravitas/AutoGPT/pull/15619), [#15639](https://github.com/Significant-Gravitas/AutoGPT/pull/15639), [#15647](https://github.com/Significant-Gravitas/AutoGPT/pull/15647))
- **Claude Haiku 5.5 and Meta Muse Spark Open** join the model catalog. ([#15307](https://github.com/Significant-Gravitas/AutoGPT/pull/15307), [#15566](https://github.com/Significant-Gravitas/AutoGPT/pull/15566))
- **New Notion blocks** update a page's title, properties, icon and cover, move it to the trash or back, and append content to it. ([#15861](https://github.com/Significant-Gravitas/AutoGPT/pull/15861))
- **HubSpot joins the MCP catalog,** and Conductor blocks return a compact reply by default. ([#15594](https://github.com/Significant-Gravitas/AutoGPT/pull/15594), [#15107](https://github.com/Significant-Gravitas/AutoGPT/pull/15107))
- **YouTube embeds wait for marketing-cookie consent.** You can open the video on YouTube instead; self-hosted installs without a consent banner offer a per-video **Load video** button. ([#15751](https://github.com/Significant-Gravitas/AutoGPT/pull/15751))

</details>

<details>

<summary>🎨 UI/UX Improvements</summary>

- **Pages open straight into the sidebar layout,** without first flashing the old top navigation. ([#15106](https://github.com/Significant-Gravitas/AutoGPT/pull/15106))
- **Without an active plan,** uploading files, connecting services, scheduling runs and creating workflows show **See plans** instead of an error. ([#15676](https://github.com/Significant-Gravitas/AutoGPT/pull/15676))
- **Checkout buttons stay busy until Stripe opens,** so a second click cannot start a second checkout. ([#15603](https://github.com/Significant-Gravitas/AutoGPT/pull/15603))
- **Screens say Otto, your Expert or workflows** where they still said AutoPilot, Copilot or agents. ([#15704](https://github.com/Significant-Gravitas/AutoGPT/pull/15704), [#15768](https://github.com/Significant-Gravitas/AutoGPT/pull/15768), [#15590](https://github.com/Significant-Gravitas/AutoGPT/pull/15590))
- **Tooltips no longer block the buttons under them,** and the browser tab title no longer carries a "New activity" counter. ([#14370](https://github.com/Significant-Gravitas/AutoGPT/pull/14370), [#15568](https://github.com/Significant-Gravitas/AutoGPT/pull/15568))
- **Shared Marketplace links preview with the listing's one-line summary,** and LinkedIn and TikTok are spelled correctly in block names. ([#14496](https://github.com/Significant-Gravitas/AutoGPT/pull/14496), [#15504](https://github.com/Significant-Gravitas/AutoGPT/pull/15504))
- **Adding a URL that is not an MCP server** shows a one-line explanation instead of that site's error page. ([#15232](https://github.com/Significant-Gravitas/AutoGPT/pull/15232))

</details>

<details>

<summary>🔒 Security & Privacy</summary>

- **Sign-up needs an address that can receive mail:** disposable domains and domains that accept no mail are refused. ([#15824](https://github.com/Significant-Gravitas/AutoGPT/pull/15824), [#15959](https://github.com/Significant-Gravitas/AutoGPT/pull/15959))
- **Sign-in endpoints we don't use are now switched off.** ([#15967](https://github.com/Significant-Gravitas/AutoGPT/pull/15967))
- **The integration setup wizard returns you only to an address the connecting app registered,** and apps connected to AutoGPT can use each refresh token once. ([#15363](https://github.com/Significant-Gravitas/AutoGPT/pull/15363), [#15632](https://github.com/Significant-Gravitas/AutoGPT/pull/15632))
- **People who don't own a workflow see its shared version,** without the owner's private settings, and every run stays in an organization its runner belongs to. ([#15906](https://github.com/Significant-Gravitas/AutoGPT/pull/15906), [#15982](https://github.com/Significant-Gravitas/AutoGPT/pull/15982))
- **Requests are checked earlier:** signed-in routes check your sign-in before reading an upload, webhook deliveries are verified before anything else runs, and outbound requests from Ollama blocks and browser tools are checked against the platform's address rules on every hop, IPv6 included. ([#15969](https://github.com/Significant-Gravitas/AutoGPT/pull/15969), [#15965](https://github.com/Significant-Gravitas/AutoGPT/pull/15965), [#15986](https://github.com/Significant-Gravitas/AutoGPT/pull/15986), [#15859](https://github.com/Significant-Gravitas/AutoGPT/pull/15859), [#15926](https://github.com/Significant-Gravitas/AutoGPT/pull/15926))
- **Less personal data leaves the app:** error reports no longer carry your email or IP address, session replays hide on-screen text by default, and onboarding voice recordings are deleted once transcribed. ([#15705](https://github.com/Significant-Gravitas/AutoGPT/pull/15705), [#15485](https://github.com/Significant-Gravitas/AutoGPT/pull/15485))
- **Consent comes first:** the feedback form and DataFast load only after you agree, Google Consent Mode starts with every signal set, and signed-out pages have a **Cookie settings** link. ([#15750](https://github.com/Significant-Gravitas/AutoGPT/pull/15750), [#15752](https://github.com/Significant-Gravitas/AutoGPT/pull/15752), [#15936](https://github.com/Significant-Gravitas/AutoGPT/pull/15936), [#15023](https://github.com/Significant-Gravitas/AutoGPT/pull/15023))

</details>

<details>

<summary>🐛 Bug Fixes</summary>

- **A message is never silently dropped:** repeating your last message sends it again, and a message sent right after **Stop** stays on screen. ([#15918](https://github.com/Significant-Gravitas/AutoGPT/pull/15918), [#15931](https://github.com/Significant-Gravitas/AutoGPT/pull/15931))
- **Messages that start with `/` or mention `@` reach Otto exactly as you wrote them,** and subtasks or to-dos that mention `sudo` are no longer refused. ([#15910](https://github.com/Significant-Gravitas/AutoGPT/pull/15910), [#15562](https://github.com/Significant-Gravitas/AutoGPT/pull/15562))
- **Files sent from an Expert's side-panel chat reach the Expert,** and your message and attachments appear immediately. ([#15582](https://github.com/Significant-Gravitas/AutoGPT/pull/15582))
- **A dropped connection while starting a chat** no longer leaves an empty chat behind. ([#15080](https://github.com/Significant-Gravitas/AutoGPT/pull/15080))
- **After switching accounts,** the browser stops showing the previous account's notifications, and **Needs you** on Home updates as soon as you answer a question. ([#15567](https://github.com/Significant-Gravitas/AutoGPT/pull/15567), [#15501](https://github.com/Significant-Gravitas/AutoGPT/pull/15501))
- **Onboarding:** a long brain dump shows its 20,000-character limit, a recording the quality check could not judge can be retried, **Skip** moves on at the first click, a slow connection no longer reports an update "in another session", the brain dump fits short screens, and a translated checkout button no longer crashes the page. ([#15960](https://github.com/Significant-Gravitas/AutoGPT/pull/15960), [#15932](https://github.com/Significant-Gravitas/AutoGPT/pull/15932), [#15927](https://github.com/Significant-Gravitas/AutoGPT/pull/15927), [#15814](https://github.com/Significant-Gravitas/AutoGPT/pull/15814), [#15610](https://github.com/Significant-Gravitas/AutoGPT/pull/15610), [#15612](https://github.com/Significant-Gravitas/AutoGPT/pull/15612))
- **Account emails:** payment emails state the right amounts, dates and next steps and fit phone screens, notification mail goes only to addresses you have verified, and an out-of-office reply no longer marks your address unverified. ([#15552](https://github.com/Significant-Gravitas/AutoGPT/pull/15552), [#15974](https://github.com/Significant-Gravitas/AutoGPT/pull/15974), [#15981](https://github.com/Significant-Gravitas/AutoGPT/pull/15981))
- **Billing:** a failed renewal affects only the subscription it belongs to, and wallet payments can resume afterwards. ([#15345](https://github.com/Significant-Gravitas/AutoGPT/pull/15345))
- **Block charges are consistent:** steps run by the Orchestrator are charged like any other step, blocks priced per provider (Exa, AgentMail, Airtable and ten more) are charged in every run and only on the platform's keys, and Otto's block runs are priced with the credentials they use. ([#15631](https://github.com/Significant-Gravitas/AutoGPT/pull/15631), [#15092](https://github.com/Significant-Gravitas/AutoGPT/pull/15092), [#15633](https://github.com/Significant-Gravitas/AutoGPT/pull/15633))
- **Workflow steps resumed after review are not charged twice.** ([#15327](https://github.com/Significant-Gravitas/AutoGPT/pull/15327))
- **Japanese, Chinese and Korean input works in question cards and JSON editors.** Editing in the middle of a value no longer jumps the cursor or discards unfinished text. ([#15649](https://github.com/Significant-Gravitas/AutoGPT/pull/15649))
- **Public run links include outputs entered directly in the Builder,** matching the owner's output view. ([#15654](https://github.com/Significant-Gravitas/AutoGPT/pull/15654))
- **Code steps accept variables up to 10 MB,** and a one-time RSS read finishes as soon as it has its entries. ([#15129](https://github.com/Significant-Gravitas/AutoGPT/pull/15129), [#15178](https://github.com/Significant-Gravitas/AutoGPT/pull/15178))
- **Deleting a chat no longer waits for background cleanup.** ([#15579](https://github.com/Significant-Gravitas/AutoGPT/pull/15579))
- **Long filenames containing multibyte characters upload safely,** and blank or very long Marketplace searches no longer break the results page. ([#15653](https://github.com/Significant-Gravitas/AutoGPT/pull/15653), [#15587](https://github.com/Significant-Gravitas/AutoGPT/pull/15587))

</details>
