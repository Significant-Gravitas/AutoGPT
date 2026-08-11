# Shipping the redesigned email system

The code in this package is complete. What is *not* in this repo — and cannot
be — is the configuration that lives in MailerLite, Postmark, Stripe, DNS and
the CDN. This file is that list. Nothing below is optional; several items will
cause visible customer-facing bugs if skipped.

Ordered by what blocks what.

---

## 1. CDN — hero art (blocks: every email)

`backend/notifications/assets/` holds the hero art and the logo. They are
hosted images rather than inline SVG because **Outlook does not render inline
SVG at all**, and rather than data URIs because **Gmail does not display them**.

- Upload the whole of `backend/notifications/assets/` to the CDN.
- Set `EMAIL_ASSET_BASE_URL` to the base URL it is served from (default
  `https://cdn.agpt.co/email`), so `hero-briefing-clean.jpg` resolves at
  `$EMAIL_ASSET_BASE_URL/hero-briefing-clean.jpg`.
- Serve with long cache headers and CORS off — these are `<img src>` loads.
- `mark-otto.png` is used only by the internal Ops band; it must be there too.

Until this is done every email renders with the correct background colour bands
(each hero `<td>` carries a sampled `bgcolor`) but no art.

## 2. Postmark — streams and senders (blocks: everything)

Alerts, Briefings and account mail all go out on a **transactional stream**,
separate from marketing mail, on their own sending domain and reputation.

- Create/confirm a transactional message stream and set
  `POSTMARK_TRANSACTIONAL_STREAM` to its ID.
- Verify the three sender signatures and their DKIM/SPF/DMARC records:
  - `billing@agpt.co` — subscription and account service messages
  - `notify@agpt.co` — the Briefing, the Alert, the Verdict
  - `platform@agpt.co` — internal ops mail to the refunds team
  Set `BILLING_SENDER_EMAIL`, `PRODUCT_SENDER_EMAIL`, `OPS_SENDER_EMAIL`.
- `REFUND_NOTIFICATION_EMAIL` is the refunds team address that Ops mail is
  delivered to.
- One-click List-Unsubscribe headers are set by the code on every family
  **except Ops**, which is internal and deliberately not opt-in. Nothing to
  configure, but do not add a stream-level unsubscribe footer to the ops
  stream.

## 3. Stripe — dunning, and turning off Stripe's own emails

The payment emails read their facts off Stripe rather than keeping a countdown
of our own, so Stripe's settings *are* the copy.

- **Turn off Stripe's built-in failed-payment and dunning emails.** If you
  don't, customers get both ours and Stripe's.
- Configure Smart Retries / automatic collection. The number of retries and the
  final date must match what `final_notice` promises: the email's "plan pauses
  on" date is taken from the invoice's `period_end`, and the copy says the
  attempts ran "over the past two weeks".
- **Decide the end state for an unpaid subscription** (cancel vs. mark unpaid)
  in Stripe's subscription settings. The "plan ended" email fires on
  `customer.subscription.deleted`; if you choose "mark unpaid", that event never
  arrives and the email never sends.
- Webhook endpoint must be subscribed to: `checkout.session.completed`,
  `customer.subscription.updated`, `customer.subscription.deleted`,
  `invoice.payment_failed` (and the newer `invoice_payment.payment_failed`).
  These are already handled in `backend/api/features/v1.py`.
- **No trial events.** `customer.subscription.trial_will_end` is deliberately
  not listened for: the platform does not offer a trial. The design document
  includes a trial email, and `templates/lifecycle.html.j2` still carries its
  branch, but nothing in the backend ever emits `kind='trial_ending'`. If
  trials are ever introduced, wiring that branch up is the whole job.

## 4. MailerLite — the tour and the changelog

Two emails live entirely in MailerLite and need no deploy to change: the
six-email **White Glove Tour** and the **monthly changelog**. The backend only
manages who is in each audience.

### 4a. The onboarding tour

- The automation **already exists** as "Subscription Onboarding — White Glove
  Tour", currently inactive. **Activate it; do not rebuild it.** The work is
  activation plus swapping in the unified design.
- Re-skin its six emails to `templates/onboarding.html.j2` (kept in this repo
  purely as the design reference — the backend never renders it). Content is
  unchanged; days 0, 2, 4, 7, 10, 14.
- **Automation re-entry must stay OFF.** Someone who cancels and returns must
  not restart the tour from part 1.
- **Reply-to is `john.ababseh@agpt.co`** and the copy promises he reads every
  one. If that stops being true, change the copy before it stops being true.
- **Day 7 is deliberately plain** — no buttons, no screenshots. Resist adding a
  CTA to it.
- **The automation's final step must move the subscriber from the tour group
  into the changelog group.** That handoff is what implements the suppression
  rule: mid-tour users simply aren't in the changelog group, so there are no
  campaign-level exclusion rules to forget. The backend deliberately does not
  touch this edge — if both sides managed it we would double-add.
- Set `MAILERLITE_ONBOARDING_GROUP_ID` to the tour group's ID. Joining that
  group is the automation's trigger.

### 4b. The monthly changelog

- Create the `changelog-subscribers` group and set
  `MAILERLITE_CHANGELOG_GROUP_ID`.
- Build the campaign template from `templates/changelog.html.j2` (again, design
  reference only — never rendered by the backend).
- One edition a month, same week each month, dated in the name
  ("AutoGPT Update — July 2026"). Predictability is the product; nothing else
  ships between editions.
- **Entries come from the public changelog** at `agpt.co/docs/platform/changelog`
  — never announce something that isn't in it.
- Sender `hello@news.agpt.co`, honours unsubscribe, one-click List-Unsubscribe
  headers. That subdomain needs its own DKIM/SPF/DMARC.

### 4c. Credentials

- Set `MAILERLITE_API_TOKEN`. Without it, audience changes fail loudly and the
  queued job retries — enrolment is never silently dropped, but it also never
  happens.

### Who owns which transition

| Transition | Owner |
| --- | --- |
| First-time subscriber → tour group | Backend (`AudienceAction.ENROLL_TOUR`) |
| Tour finisher → changelog group | **MailerLite automation's final step** |
| Returning customer / pre-tour user → changelog group | Backend (`ADD_CHANGELOG`) |
| Churned customer → out of changelog group | Backend (`REMOVE_CHANGELOG`) |

## 5. Database

- Run `poetry run prisma migrate deploy`. The migration
  `20260811120000_replace_email_notification_system` recreates the
  `NotificationType` enum, so it **truncates `NotificationEvent` and
  `UserNotificationBatch`** — queued payloads in the old shape cannot be
  rendered by any current template. Drain or accept the loss of anything
  sitting in those tables before deploying.
- The old per-type preference columns are dropped. The migration carries intent
  across first: a user who had the weekly summary off keeps their Briefing off,
  and a user who had every balance notification off keeps Alerts off.

## 6. Queues

The RabbitMQ queues are renamed to `_v3` (`user_notifications_v3`,
`ops_notifications_v3`, `audience_changes_v3`, `failed_notifications_v3`). The
old `_v2` queues are left in place to drain under old-image consumers and can
be deleted once empty.

## 7. Deliberately left in place

`NotificationEvent` and `UserNotificationBatch` survive as empty tables. The
batching they existed for went with the per-run email, and no code in this
package touches them any more — but they are also wired into the org-tenancy
migration tooling (`backend/data/org_migration.py`), which is a different
subsystem with its own tests. Dropping them is a small follow-up, not part of
the email change.

## 8. Behaviour change to be aware of

`queue_notification` / `queue_notification_async` used to short-circuit in
production and return success **without publishing**, so no notification email
was actually sent in production. The replacement does not do that: once this
ships with Postmark configured, these emails really send. That is the point of
the change, but it is the single largest operational difference and worth
staging first.

## 9. Still open

- **Two admins must not refund the same person twice.** Ops mail goes to the
  whole refunds team at once, so two people can open it and act on the same
  request within seconds of each other. The email now deep-links into the admin
  panel (rather than pasting a Supabase console URL and table-editing steps, as
  the old template did), but the admin panel still needs a robust guard:
  claiming a request, a status that locks once a refund is issued, and an
  idempotency key on the Stripe call, so the second attempt fails loudly
  instead of double-refunding. This is an admin-panel change, not an email one.
- **Discrete review feedback.** The Verdict's "changes requested" variant
  renders a numbered list when the review supplies `changes` as discrete items,
  and falls back to free-text `comments` otherwise. The store review UI
  currently only collects free text; collecting discrete items is what the
  rendering was designed for.
- **Starred runs** are listed in the design as an interestingness signal. The
  platform has no way to star a run, so `compute_score` does not use one; add
  the signal when the feature exists.
- **Four alert causes have no producer yet.** The Alert family renders all nine
  causes in the catalog, and four are wired to real signals today
  (`low_balance`, `zero_balance`, `awaiting_review`, and anything raised via
  `raise_alert`). The remaining five describe platform states that the platform
  does not currently detect, so nothing calls them:

  | Cause | What has to exist first |
  | --- | --- |
  | `auth_expired` | A hook on integration-credential refresh failure, raising the condition and resolving it on reconnect |
  | `paused_failures` | Agents do not auto-pause after repeated failures today; that is a product feature, not an email one |
  | `block_failed` | Per-block repeated-failure detection per agent (the old `BLOCK_EXECUTION_FAILED` type had no producer either) |
  | `continuous_error` | Multi-day consecutive-failure detection (same — the old `CONTINUOUS_AGENT_ERROR` type never fired) |
  | `awaiting_input` | A run state for "waiting on a value for this field", distinct from waiting on approval |
  | `guardrail` | A user-facing spend limit; the existing expert weekly budget is not surfaced as one |

  Wiring any of them is one call to `backend.notifications.alerts.raise_alert`
  with the matching cause model, and one `resolve_alert` when the condition
  clears. The email side needs no further work.
