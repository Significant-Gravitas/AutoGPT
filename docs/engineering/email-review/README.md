# Transactional email comparisons

These are full-page browser screenshots of real Jinja-rendered emails, with synthetic names, amounts and links. Left is the integration base (billing PR #15345 plus dev); right is the email follow-up. No customer messages, private review documents, account identifiers or production credentials are included.

- Before source: `f8fee36faa38e2efdeaf51b9bc30ac82b7db27e2`.
- After source: `89ffd34bf36cf28705f7450aa31ebf9aa45001c5`.
- 52 variants: 28 shared design fixtures, 10 cause-generated alerts, 4 auth messages, 7 marketing reference pages, and 3 conditional billing/greeting variants. The marketing references are not sent by the application.
- Every variant has a 390px comparison. Twelve representative variants also have 640px desktop and 390px stripped-style comparisons. Each side is captured at its stated viewport width, without scaling; only a label bar and a 16px separator were added.
- The Briefing estimates are produced by each revision's actual `build_briefing` with fake RPC results. Balance forecasts and shortfall strings execute each revision's actual formatter. This exposes the original double conversion rather than hiding it behind preview-only sample text.
- Both revisions use the repository's shared scenario fixtures. The after fixtures include the new last-attempt and actual conversion dates, matching the payload changes. Plan prices are illustrative test values.
- Image requests retain their original URLs and are served with the exact repository PNG bytes. The Otto and wordmark bytes were checked against the live public URLs and match. The declared Google Fonts stylesheets/fonts were loaded when available.
- 728 browser cases cover both revisions at 640, 390 and 360px, stripped styles at 390 and 360px, images blocked at 390px, and a dark color preference at 390px. After cases have no horizontal overflow or unexpected broken images. The inline content measures 346px at a 390px viewport, including with style blocks removed.
- A browser's dark preference is a desk check, not a claim about Gmail/Outlook's proprietary recoloring. Actual inbox and phone validation remains a release check. The isolated browser context was closed and its temporary profile removed.

[Measurements](measurements.json) record the browser checks. The comparison PNGs below preserve both complete emails.

| Variant | 390px | Desktop | Styles removed |
|---|---|---|---|
| alert-0 | [Before / after](alert-0-390-normal.png) | — | — |
| alert-3 | [Before / after](alert-3-390-normal.png) | — | — |
| auth-change_email | [Before / after](auth-change_email-390-normal.png) | — | — |
| auth-reset_password | [Before / after](auth-reset_password-390-normal.png) | — | — |
| auth-set_password | [Before / after](auth-set_password-390-normal.png) | — | — |
| auth-verify_email | [Before / after](auth-verify_email-390-normal.png) | [Before / after](auth-verify_email-640-normal.png) | [Before / after](auth-verify_email-390-style-off.png) |
| briefing-attention-overflow | [Before / after](briefing-attention-overflow-390-normal.png) | — | — |
| briefing-daily | [Before / after](briefing-daily-390-normal.png) | — | — |
| briefing-monthly | [Before / after](briefing-monthly-390-normal.png) | — | — |
| briefing-quiet | [Before / after](briefing-quiet-390-normal.png) | — | — |
| briefing-rough | [Before / after](briefing-rough-390-normal.png) | — | — |
| briefing-weekly | [Before / after](briefing-weekly-390-normal.png) | [Before / after](briefing-weekly-640-normal.png) | [Before / after](briefing-weekly-390-style-off.png) |
| cause-auth-expired | [Before / after](cause-auth-expired-390-normal.png) | — | — |
| cause-awaiting-input | [Before / after](cause-awaiting-input-390-normal.png) | — | — |
| cause-awaiting-review | [Before / after](cause-awaiting-review-390-normal.png) | — | — |
| cause-block-failed | [Before / after](cause-block-failed-390-normal.png) | — | — |
| cause-continuous-error | [Before / after](cause-continuous-error-390-normal.png) | — | — |
| cause-guardrail | [Before / after](cause-guardrail-390-normal.png) | — | — |
| cause-low-balance | [Before / after](cause-low-balance-390-normal.png) | [Before / after](cause-low-balance-640-normal.png) | [Before / after](cause-low-balance-390-style-off.png) |
| cause-low-balance-no-forecast | [Before / after](cause-low-balance-no-forecast-390-normal.png) | — | — |
| cause-paused-failures | [Before / after](cause-paused-failures-390-normal.png) | — | — |
| cause-zero-balance | [Before / after](cause-zero-balance-390-normal.png) | [Before / after](cause-zero-balance-640-normal.png) | [Before / after](cause-zero-balance-390-style-off.png) |
| ended-payment-False | [Before / after](ended-payment-False-390-normal.png) | — | — |
| ended-payment-True | [Before / after](ended-payment-True-390-normal.png) | — | — |
| ops-processed | [Before / after](ops-processed-390-normal.png) | — | — |
| ops-request | [Before / after](ops-request-390-normal.png) | [Before / after](ops-request-640-normal.png) | [Before / after](ops-request-390-style-off.png) |
| payment_failed | [Before / after](payment_failed-390-normal.png) | [Before / after](payment_failed-640-normal.png) | [Before / after](payment_failed-390-style-off.png) |
| payment_final_notice | [Before / after](payment_final_notice-390-normal.png) | [Before / after](payment_final_notice-640-normal.png) | [Before / after](payment_final_notice-390-style-off.png) |
| reference-changelog | [Before / after](reference-changelog-390-normal.png) | [Before / after](reference-changelog-640-normal.png) | [Before / after](reference-changelog-390-style-off.png) |
| reference-tour-1 | [Before / after](reference-tour-1-390-normal.png) | [Before / after](reference-tour-1-640-normal.png) | [Before / after](reference-tour-1-390-style-off.png) |
| reference-tour-2 | [Before / after](reference-tour-2-390-normal.png) | — | — |
| reference-tour-3 | [Before / after](reference-tour-3-390-normal.png) | — | — |
| reference-tour-4 | [Before / after](reference-tour-4-390-normal.png) | — | — |
| reference-tour-5 | [Before / after](reference-tour-5-390-normal.png) | — | — |
| reference-tour-6 | [Before / after](reference-tour-6-390-normal.png) | — | — |
| subscription_cancelled | [Before / after](subscription_cancelled-390-normal.png) | — | — |
| subscription_resumed | [Before / after](subscription_resumed-390-normal.png) | — | — |
| subscription_welcome | [Before / after](subscription_welcome-390-normal.png) | [Before / after](subscription_welcome-640-normal.png) | [Before / after](subscription_welcome-390-style-off.png) |
| trial-canceled | [Before / after](trial-canceled-390-normal.png) | — | — |
| trial-converted | [Before / after](trial-converted-390-normal.png) | [Before / after](trial-converted-640-normal.png) | [Before / after](trial-converted-390-style-off.png) |
| trial-ended | [Before / after](trial-ended-390-normal.png) | — | — |
| trial-ending | [Before / after](trial-ending-390-normal.png) | — | — |
| trial-payment_failed | [Before / after](trial-payment_failed-390-normal.png) | — | — |
| trial-resumed | [Before / after](trial-resumed-390-normal.png) | — | — |
| trial-started | [Before / after](trial-started-390-normal.png) | [Before / after](trial-started-640-normal.png) | [Before / after](trial-started-390-style-off.png) |
| verdict-approved | [Before / after](verdict-approved-390-normal.png) | — | — |
| verdict-changes | [Before / after](verdict-changes-390-normal.png) | — | — |
| verdict-no-feedback | [Before / after](verdict-no-feedback-390-normal.png) | — | — |
| welcome-experts-enabled | [Before / after](welcome-experts-enabled-390-normal.png) | — | — |

### Conditional billing and greeting cases

These additional payment payloads are produced by the actual lifecycle handler with fake Stripe/user/queue boundaries. They show a `past_due` subscription without a retry date, an `unpaid` subscription with a hosted invoice, and an unnamed account.

- [greeting-fallback: before / after](greeting-fallback-390-normal.png)
- [payment-no-retry-date: before / after](payment-no-retry-date-390-normal.png)
- [final-hosted-invoice: before / after](final-hosted-invoice-390-normal.png)
