# Deleting an account (privacy requests)

Deleting an account has two halves:

- **Closing** (SECRT-2877, `backend/api/features/account_deletion/service.py`): the person deletes their account from Settings › Account or the paywall. Closing cancels their Stripe plan now, disconnects and revokes their apps, revokes their API keys and OAuth grants, stops their schedules, triggers, AutoPilot turns and runs, switches off their email, bans sign-in, ends every session, takes their Marketplace listings down, writes an `AccountDeletionRequest` row and sends the "Your account is deleted" email. It deletes no data.
- **Erasure** (SECRT-2854, `backend/data/account_erasure/`): this page. It processes every `AccountDeletionRequest` whose `completedAt` is still null and removes the account's data, then sets `completedAt`.

A request sent to privacy@agpt.co goes through the same two halves: the `erase-account --close` CLI closes the account exactly as the Settings button does (with source `EMAIL`), then erases it.

We promise to delete within 30 days of the request.

## How erasure runs

Two ways, the same routine (`routine.erase_account`):

- **The periodic job**, `erase_closed_accounts`, every hour (`ACCOUNT_ERASURE_SWEEP_INTERVAL_SECS`). It is **off by default**: set `ACCOUNT_ERASURE_SWEEP_ENABLED=true` on the scheduler to turn it on. The scheduler has no database connection, so the job only starts the sweep over RPC; the sweep itself runs inside the **database manager** (`start_account_erasure_sweep`), like the PostHog lifecycle sweep. It erases every account closed at least 15 minutes ago and not erased yet, oldest first; one that is blocked or fails is logged and left for the next run.
- **The CLI**, `poetry run cli erase-account`, for one account or every pending one. Dry run unless `--apply`.

### Before turning the job on

The database manager runs every step, so **the database manager pod needs every setting the erasure uses**, which today only the backend pods have:

- **Buckets**: `MEDIA_GCS_BUCKET_NAME`, `PUBLIC_SITE_MEDIA_BUCKET`, `PRIVATE_USER_DATA_BUCKET` and the GCS credentials to delete from them. Without any bucket configured the files step falls back to the pod's local data directory, which on a cloud stack finds nothing and leaves the account's files in the buckets.
- **FalkorDB**: `GRAPHITI_FALKORDB_HOST`, `GRAPHITI_FALKORDB_PORT`, `GRAPHITI_FALKORDB_PASSWORD`. With the host empty the memory step is skipped.
- **PostHog**: `POSTHOG_PERSONAL_API_KEY` (a personal API key with the `person:write` scope), `POSTHOG_PROJECT_ID`, and `POSTHOG_APP_HOST` if the project is not in the EU cloud. Without them the PostHog step is skipped.
- **MailerLite**: `MAILERLITE_API_TOKEN`. Without it the MailerLite step is skipped.
- **Stripe**: `STRIPE_API_KEY`, for the subscriptions of orgs that go with the account.
- It also reaches the scheduler and the notification manager over RPC, and Redis and RabbitMQ (the work stop), as the other services do.

A skipped step does not stop the erasure: the request is still marked done, and the sweep logs a warning naming the skipped steps. Check the database manager's logs after the first runs; a skip there means a setting is missing.

## Who runs the CLI, and where

The production operator, from a shell in a production backend pod: the CLI needs the production database, Redis, RabbitMQ, the scheduler and notification services, FalkorDB, the buckets, Stripe, PostHog and MailerLite, which only those pods reach. `POSTHOG_APP_HOST` defaults to the EU cloud (`https://eu.posthog.com`); a project in the US cloud sets `https://us.posthog.com`. Each cloud refuses the other's keys, so a wrong host fails the step instead of finding no person.

The memory step is skipped only when `GRAPHITI_FALKORDB_HOST` is empty, which is how a stack without FalkorDB says so. Where it is set (production and dev set it in the shared config), a FalkorDB the pod cannot reach fails the step, and the erasure stops before the account's rows go; fix the connection and run it again. A self-hosted stack that does not run FalkorDB sets `GRAPHITI_FALKORDB_HOST=` (empty), since the default points at `localhost`.

Every command is a dry run unless it has `--apply`, and prints counts only, never the email. Every step is idempotent: a failed `--apply` stops with the account's rows still in place, and running the same command again resumes.

## A request sent by email

1. **Check the request.** It must come from the account's own email address, or prove control of it (reply from that address). Never act on a request about someone else's account.
2. **Check with the owner whether a refund is due, before closing.** Closing cancels the account's subscriptions at once, and nothing refunds anything. Whether, when and how much to refund is an open business decision; until it is made, ask the owner for each request. Make any refund in Stripe by hand.
3. **Close and erase.** Pass the email or the user ID. Dry run first:

   ```sh
   poetry run cli erase-account sam@example.com --close
   poetry run cli erase-account sam@example.com --close --apply
   ```

   `--close` closes the account the way Settings does, with source `EMAIL`, then the erasure runs. The confirmation email is the one Settings sends, with its own reason line: "You asked us by email to delete your account." Closing also sends the anonymous `account_deletion_completed` count, with `surface: email`, which the tracking plan does not list yet. If Stripe refuses the plan cancel, nothing is closed or erased: run it again. An account already closed (from Settings, say) is only erased; `--close` changes nothing then. Like Settings, `--close` refuses an account banned for another reason (abuse, say), before it touches Stripe or anything else.

   It refuses while a blocker remains, and nothing is deleted then:
   - `not_closed`: the account has no deletion request. Pass `--close` for a request made by email.
   - `shared_orgs_without_another_owner`: the account is the only owner of an organization with other active members. Ask the requester to make someone else an owner, or decide with the org's members. Invited, suspended and removed members do not count: an organization only they are left in goes with the account, and its subscription is cancelled. The job logs this account at every run until it is resolved.
   - `oauth_apps_in_use`: the account registered an OAuth app (it is the app's developer) that another account still holds a live grant to. Erasing the account would delete the app and cut those accounts off, and the app is not handed to the "deleted user" account: the developer still holds its client secret and controls its redirect URIs, and nobody here could manage it. Decide with the product team, then either move the app to another owner (set `OAuthApplication.ownerId` to their user ID) or, if it should stop working, disable it (`isActive` = false). A disabled app, or one whose grants are all revoked or expired, goes with the account. The job logs this account at every run until it is resolved.
   - `banned`: with `--close`, the account is banned for a reason other than closing it. Check with whoever banned it. To close and erase it anyway, add `--allow-banned`: the ban keeps its reason and expiry.
   - `invalid_account_id`: the ID is not a user ID. Nothing runs.
   - `system_account`: the "deleted user" placeholder (below) is never erased.

   It exits non-zero unless every step finished.
4. **Confirm.** Run the dry run again with the user ID (the email no longer resolves) and check every count is 0. It reports the account as not found, which is expected now.
5. **Follow up by hand** on what the routine does not cover (below), then reply to the requester that their account and data are deleted. Record the request and its dates wherever privacy requests are tracked, by user ID; do not paste the email into tickets or channels.

To erase every closed account by hand (when the job is off, or to see what it would do):

```sh
poetry run cli erase-account --pending
poetry run cli erase-account --pending --apply
```

The apply erases the accounts the dry run listed as unblocked.

## What it does, in order

The account's rows go last, because every earlier step finds its data through them; a step that fails stops the erasure there, and the next run (or a rerun of the CLI) resumes.

1. **Work** (`stop_work`): reruns closing's `stop_user_work`. A schedule or trigger whose removal failed when the account was closed is not retried otherwise, and neither the scheduler nor the webhook ingress checks for a closed account.
2. **Confirmation**: if closing could not send the account-deleted email (`confirmationSentAt` is null), sends it now, while the address still exists, and records it. A send that fails stops the erasure, so the next run tries again.
3. **Org subscriptions**: expires the open Checkouts and cancels the subscriptions of the organizations that go with the account, when their Stripe customer is not the account's own (closing cancelled that one).
4. **Shared agents**: hands the agents other accounts use to the "deleted user" account (below), with their marketplace listings.
5. **Marketplace**: its reviews, its other listings (which closing took down) with their versions and search embeddings, and every skill listing it published, with its versions and files. A skill listing is never handed over: an install copies the skill into the installer's own skills folder, so people who installed it keep their copy. Left to the account row's delete, the listing would stay with no owner, which marks a skill the platform wrote.
6. **Files**: everything under its prefixes in every bucket (private, public and legacy): `workspaces/{workspace id}/` (workspace files), `workspaces/cli-sessions/{user id}/` (AutoPilot transcripts), `uploads/users/{user id}/` (uploads, onboarding voice recordings), `users/{user id}/` (marketplace media and avatars, including their public copies) and `oauth-apps/{app id}/` for OAuth apps it owns. A self-hosted stack's folders on disk likewise. Media a listing that stays still shows is kept (below).
7. **Memory**: AutoPilot's memory graphs in FalkorDB, Otto's and one per expert it owns. Skipped on a stack with no FalkorDB host configured (above).
8. **PostHog**: the person, with its events and recordings (PostHog deletes events in a weekly batch).
9. **MailerLite**: "forgets" the subscriber, which removes them now and erases their data within 30 days.
10. **Database**: first the account's own runs, in batches. Then, in one transaction: strips the personal content from its credit ledger (kept, below); deletes its refund requests, private search embeddings, node key-value data, stored credentials, pending invitations to it, its webhook rows, the organizations only it belongs to (personal org included, with their ledger, teams, credentials and audit log), its auth identity with sessions and sign-in accounts, and the account row, which cascades to everything else it owns: its other agents, chats, library, experts, workspace records, API keys, OAuth apps (none another account still uses: that blocks the erasure, above) and tokens, notifications, onboarding and the rest. The identity goes in the same transaction as the account, so the orphan-healing job (`heal_orphaned_auth_identities`) never sees an identity without an account and re-creates it.
11. **Request**: sets the request's `completedAt`. The row itself stays, with the user ID, so backend auth keeps refusing a token issued before the account was closed, and nothing re-creates the account from it.

Chats and runs the account created inside a shared organization belong to the account and go with it. Its agents there stay, since the org's members can open them (below).

## Agents other accounts use

Deleting an agent another account uses would take that account's library entry, presets and runs with it. So instead of deleting it, the erasure hands it to a placeholder account and the agent keeps working for everyone using it: their library entries, presets, schedules, triggers and runs stay as they are.

An agent counts as used by others when another account has it in its library or presets, has run it, lists it in the marketplace, or an expert of theirs (or a roster template) installed its listing. Its own listing alone does not count: closing the account took every listing of its own down, so nobody can add the agent any more, and one nobody has added is deleted. So does an agent in an organization someone else is an active member of, at the org's home or in one of its teams, since every member of the org (or of the team) can open it without adding it anywhere; it keeps its org and team. Every version of such an agent goes over, its unpublished versions included, since library entries and runs pin versions. So does every agent it runs as a sub-agent, at any depth, because a sub-agent only loads when it has the same owner as the agent at the top.

The placeholder is a platform account with ID `d83236d9-19c5-455f-90c8-4cc6d76d4dac`, email `deleted-user@agpt.invalid` and a marketplace profile named "Deleted user" (handle `deleted-user`). It has no auth identity, so nobody can sign in as it, and every email and marketing setting is off. The first erasure that needs it creates it; the routine refuses to erase it.

What changes for the agent's listing:
- It stays taken down: closing the account took every listing of its own down, and the erasure does not put it back up. It is shown as by "Deleted user" wherever it still appears (other accounts' library entries). The creator's name, avatar, bio and links go with their profile. A slug the placeholder already uses gets a number added.
- Its images and videos stay where they are, under the old account's `users/{user id}/` path, which the files step leaves alone. That applies to every listing that stays, and to the images other accounts' library entries copied from them.
- A version still waiting for review is rejected, so nothing the creator had not published gets published after they asked to be deleted.
- Reviews other accounts left stay; the creator's own reviews of other agents are deleted.
- Other accounts' library entries name "Deleted user" as the creator.

The agent's content stays as the creator last saved it, node inputs included. Credentials it referenced are deleted with the account, so they work for nobody; each account running the agent uses its own.

## What it keeps, and why

- **The credit ledger** (`CreditTransaction`): billing records are kept for six years ("Billing and payment records: 6 years" in the privacy policy, as UK tax law requires). The rows keep their user ID, amounts, types, dates, running balances and Stripe IDs. Their metadata is cut down to IDs and amounts: the block inputs, free-text reasons, and the names, emails and addresses in stored copies of Stripe objects go (`backend/data/account_erasure/ledger.py`). The admin credit views show these rows without an email.
- **Stripe billing records**: the customer, invoices, payments and refunds stay in Stripe; the customer's metadata still holds the old user ID.
- **Agents other accounts use**, their listings and those listings' media (above).
- **Trial claims**: `SubscriptionTrialClaim` keeps hashes that stop the same person claiming a second free trial with a new account. They carry no account ID, on purpose.
- **Our cost records**: `PlatformCostLog` rows stay with their user ID set to null.
- **Audit log rows** of shared organizations the account acted in stay, with its user ID.
- **Langfuse traces** expire on their own after 90 days.
- **Database backups** age out on their retention schedule.

## Not covered: do by hand, or let expire

- **Refunds**: see "A request sent by email", step 2. An open business decision.
- **The org ledger** (`OrgCreditTransaction`) of an organization no one else is an active member of is still deleted with the org. Personal orgs bill the account's own ledger, so this only matters for a team org with one member left; whether the six-year rule applies to it as well is not decided.
- **OAuth grants at the providers** (Google, GitHub, ...): closing revokes them through the credential-deletion route's revocation, but some providers (Notion, Todoist, Airtable, ...) have no revoke endpoint, so their grant lasts until the person removes it or it expires. The erasure deletes the stored tokens.
- **E2B volumes** named `autogpt-user-{user id}` and `autogpt-expert-{expert id}` (desktop blocks). Sandboxes stop on their own timeout; volumes do not. Delete them in the E2B dashboard.
- **Mem0** memories stored with the user ID by the Mem0 blocks: delete with Mem0's `delete_all(user_id=...)`.
- **Linear** customer created for feature requests (`externalId` = user ID, named with the email).
- **Ayrshare** profile titled `User {user id}-...`: Ayrshare has no API to delete it.
- **LaunchDarkly** context and **Sentry** user (ID only): no deletion API used here.
- **Redis keys** with the user ID (rate limits, usage counters, caches, AutoPilot session state) expire on their own TTLs, the longest after 90 days.
- **A chat-platform AutoPilot turn already running** when the account was closed is not cancelled (closing's residual gaps); its chat is deleted, so a turn that finishes afterwards has nowhere to write.
