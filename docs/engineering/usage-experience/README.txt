Usage limits and plan upgrades — implementation review
======================================================

Every After panel is a browser screenshot of the actual components in this PR.
There are no independently designed mockups in this directory. The component
code is the source of truth: regenerate the images whenever that code changes.

comparisons/ contains all 33 compact before/after pairs. implementation/ contains
the actual after screenshots, including the six original review filenames.
The complete capture manifest records source hashes, screenshot hashes, visible
copy, controls, viewport dimensions, fixture data, and verified browser cleanup.
See render/README.txt for repeatable build, capture and verification commands.

The Before panels render production components from release
4473fbbcb1951471df0a05374d47c93c4e39bea6, verified from the public production
application on October 12, 2026. They use illustrative account data and deployed
CSS/fonts. These are not authenticated customer-account recordings. Where no
production screen exists, the left side is explicitly a comparison note, not
invented product UI. Details are in production-state-audit.txt.

Both sides use a fixed sample clock of October 7, 2026 at 12:00 UTC. Real reset
and payment dates use the viewer's locale/time zone. Prices, percentages, invoice
lines, dates, provider names and draft text in the pictures are fixtures. Actual
billing and usage values come from authenticated APIs. No live payment was made.

The after renderer imports the real billing page, notices, dialogs, shared chat
layout and composer, CSS, Tailwind configuration and fonts. Only external data,
navigation and inactive integrations are replaced with fixtures. No screenshot
CSS changes the product layout. Long mobile drawers are captured at sufficient
height to show the default visible content; the real 1024px dialog breakpoint is
preserved. Collapsed disclosures stay in their default collapsed state.

New trials have one lifetime allowance. Previously accepted rolling offers keep
their actual daily/weekly policy. Missing policy data falls back safely to the
legacy presentation. Internal usage-budget dollars are never shown.

Early paid activation depends on #15208, stacked above #15184. The review shows
the actual amount due, tax and billing dates, and makes the trial ending early
explicit. Readiness requires authoritative payment, entitlement and reset data,
then successful subscription/trial/usage refreshes. A recovered already-paid
plan does not promise fresh usage without an activation/reset receipt.

Paid Pro-to-Max upgrades preview Stripe's prorated debit, unused Pro credit,
tax and amount due. Confirmation reuses the reviewed proration date and checks
that commercial terms still match. An expired or changed quote requires a new
review. A failed preview offers Retry review and Keep Pro; the new UI does not
fall back to charging an undisclosed amount. Existing annual billing, scheduled
changes, provider continuation, invoices and Automation Credits remain supported.

Rollout still requires the parent activation migration/worker rollout and Stripe
test-mode validation. Local unit/interaction tests and screenshot fixtures do not
prove a real payment was completed. This PR does not alter production accounts.
