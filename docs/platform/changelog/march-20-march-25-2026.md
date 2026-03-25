# Import workflows from other platforms and enjoy a polished marketplace

*March 20 – March 25, 2026*

**Platform version:** `v0.6.53`

This release makes it easier than ever to **bring your existing automations into AutoGPT**. You can now import workflows straight from n8n, Make.com, and Zapier — plus the marketplace gets another round of visual polish.

***

## Import workflows from n8n, Make.com & Zapier

Switching to AutoGPT no longer means starting from scratch. A new **workflow import** feature lets you bring in automations you've already built on other platforms and convert them into AutoGPT agents. [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12440)

<figure><img src="../.gitbook/assets/import-workflows-hero.png" alt="Import workflows from other platforms"><figcaption><p>Import your existing workflows from n8n, Make.com, and Zapier directly into AutoGPT.</p></figcaption></figure>

***

## A more polished marketplace

The marketplace continues to get cleaner and easier to browse. Card descriptions are now **neatly truncated** to keep the layout consistent, the download button has been **repositioned** for better flow, and card overflow issues have been resolved. [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12494)

<figure><img src="../.gitbook/assets/marketplace-ui-v053-hero.png" alt="UI improvements to the marketplace"><figcaption><p>Cleaner marketplace cards with consistent layout and improved button placement.</p></figcaption></figure>

***

<details>
<summary>Improvements</summary>

* **Dry-run execution mode** — Test your agents end-to-end without making real API calls or using credits. An LLM simulates each block so you can verify wiring before going live. [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12483)
* **Parallel AutoPilot actions** — When AutoPilot needs to perform several steps at once, it now runs them simultaneously — no more waiting for each to finish before starting the next. [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12472)
* **Scoped AutoPilot tools** — You can now control exactly which tools and blocks AutoPilot has access to — whether running it as a block or using sub-agents — so you can build tightly constrained agentic systems. [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12482)
* **Leaner tool schemas** — Tool schema token cost has been reduced by 34%, meaning faster and cheaper AutoPilot. [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12398)
* **Admin marketplace preview** — Admins can now preview and download submitted agents before approving them. [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12536)

</details>

<details>
<summary>Fixes</summary>

* Fixed 5 production Sentry alerts for improved stability [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12496)
* OAuth popup detection — the app now notices when you close an OAuth window and lets you dismiss the waiting modal [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12443)
* Added circuit breaker to prevent infinite tool-call retry loops [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12499)
* Reduced noisy error logging from user-caused LLM API errors [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12516)
* Fixed browser automation to use system Chromium on all architectures [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12473)

</details>

<details>
<summary>Under the hood</summary>

* Renamed SmartDecisionMakerBlock to OrchestratorBlock for clarity [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12511)
* Added DB_STATEMENT_CACHE_SIZE env var for Prisma engine tuning [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12521)
* Bumped stagehand ^0.5.1 → ^3.4.0 to fix yanked litellm dependency [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12539)
* Registered AutoPilot sessions with stream registry for SSE updates [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12500)
* Prevented logging of sensitive data in SafeJson fallback [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12547)
* Allowed /tmp as valid path in E2B sandbox file tools [↗](https://github.com/Significant-Gravitas/AutoGPT/pull/12501)

</details>
