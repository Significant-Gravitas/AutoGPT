# Deleting an account (privacy requests)

Deleting an account has two halves:

- **Closing** (SECRT-2877, `backend/api/features/account_deletion/service.py`): the person deletes their account from Settings › Account or the paywall. Closing cancels their Stripe plan now, disconnects and revokes their apps, revokes their API keys and OAuth grants, stops their schedules, triggers, AutoPilot turns and runs, switches off their email, ends every session, takes their Marketplace listings down, writes an `AccountDeletionRequest` row and sends the "Your account is closed" email. It deletes no data, and it **does not ban sign-in**: for **14 days** (`eraseAfter` on the row) the owner can sign in and choose **Keep my account**, which deletes the row and opens the account again. A ban for another reason is left as it is; a banned person may still delete their account (it is their right), but can't sign in to keep it.
- **Erasure** (SECRT-2854, `backend/data/account_erasure/`): this page. Once the 14 days are over, it processes every `AccountDeletionRequest` whose `eraseAfter` has passed and whose `completedAt` is still null: it claims the row (sets `startedAt`), removes the account's data, then sets `completedAt`.

A request sent to privacy@agpt.co goes through the same two halves: `erase-account --close` closes the account exactly as the Settings button does (with source `EMAIL`), and the job erases it once its 14 days are over.

We promise to delete within 30 days of the request: 14 days to keep the account, then the erasure. With the hourly job switched on (it is off by default, below), the erasure runs within the hour after the 14 days; while it is off, run `erase-account --pending --apply` by hand. An erasure that is blocked or fails stays open and is not erased until someone resolves it, so check for those well before the 30 days are up.

### `startedAt`: the claim

Keep and the erasure must never both win. Keep (`reopen_account`) deletes the row only while `startedAt` is null and `eraseAfter` is ahead. The erasure claims the row before any step, in a short statement of its own: `UPDATE ... SET "startedAt" = COALESCE("startedAt", now() AT TIME ZONE 'UTC') WHERE "userId" = ... AND "completedAt" IS NULL AND "eraseAfter" <= now() AT TIME ZONE 'UTC'`. The claim reads the database's clock; Keep compares with the app server's clock. A skew between them moves the edge of the 14 days by that much, but they can't both win: each statement re-checks its condition on the locked row, and Keep deletes only a row with no `startedAt`. If the claim matches no row (the owner kept the account, the 14 days are not over, or the erasure already finished), nothing runs. Once `startedAt` is set, Keep answers "too late" (`GET /api/account/deletion` reports `too_late`).

`startedAt` is **never set back to null**. An erasure that stopped halfway (a failed step, a crashed pod) leaves the row with `startedAt` set and `completedAt` null; the next run selects it again (selection does not filter on `startedAt`), the claim keeps the first date, and the idempotent steps resume. Never clear `startedAt` by hand to "retry": that would let the owner keep an account that is already half erased.

Because a resuming run may claim again, the claim does not keep two runners apart; a Redis lease per account does (`lease.py`). Each `--apply` (the job's or the CLI's) takes it before it reads the account and holds it until the erasure ends, so two database manager replicas, or the CLI beside the job, never erase the same account at once: the second one reports "another run is erasing it" and moves on. The lease lasts 5 minutes. A live run keeps it: a heartbeat renews it every minute, since one step (the files sweep of a large account) can run for minutes, and the erasure checks it again before each step and stops if it was lost. A run that crashed or was killed renews nothing, so its lease runs out within 5 minutes and the next run resumes. Until then a rerun reports "another run is erasing it" and exits non-zero; that is the dead run's lease, not a second erasure, so wait 5 minutes and run it again. With Redis down nothing is erased.

### When an erasure stalls

An erasure that started and did not finish (a step that keeps failing, a service left unset, a CLI run that crashed and was not run again) leaves the account closed, past its 14 days and not erased. The scheduler checks for that every hour (`report_stalled_account_erasures`, `ACCOUNT_ERASURE_STALL_CHECK_INTERVAL_SECS`), whether the job is on or off: a request whose `startedAt` is over 3 hours old (`ACCOUNT_ERASURE_STALL_ALERT_SECS`) with no `completedAt` pages Discord and Sentry with its user ID, at most once a day per account. To clear it, run the dry run with that user ID (`erase-account <user ID>`) to see which step stops it, fix that, and run it again with `--apply`; the next check is quiet once `completedAt` is set.

## How erasure runs

Two ways, the same routine (`routine.erase_account`):

- **The periodic job**, `erase_closed_accounts`, every hour (`ACCOUNT_ERASURE_SWEEP_INTERVAL_SECS`). It is **off by default**: set `ACCOUNT_ERASURE_SWEEP_ENABLED=true` on the scheduler to turn it on. The scheduler has no database connection, so the job only starts the sweep over RPC; the sweep itself runs inside the **database manager** (`start_account_erasure_sweep`), like the PostHog lifecycle sweep. It erases every account whose 14 days are over and that is not erased yet, oldest first; one that is blocked or fails is logged and left for the next run, and one the owner kept in the meantime is skipped. Each run also purges the ban records past their date (below).
- **The CLI**, `poetry run cli erase-account`, for one account or every pending one, also only once the 14 days are over. Dry run unless `--apply`.

### Before turning the job on

The database manager runs every step, so **the database manager pod needs every setting the erasure uses**, which today only the backend pods have:

- **Buckets**: `MEDIA_GCS_BUCKET_NAME`, `PUBLIC_SITE_MEDIA_BUCKET`, `PRIVATE_USER_DATA_BUCKET` and the GCS credentials to delete from them. Without any bucket configured the files step falls back to the pod's local data directory, which on a cloud stack finds nothing and leaves the account's files in the buckets.
- **FalkorDB**: `GRAPHITI_FALKORDB_HOST`, `GRAPHITI_FALKORDB_PORT`, `GRAPHITI_FALKORDB_PASSWORD`. With the host empty the memory step is skipped, which stops the erasure (below).
- **PostHog**: `POSTHOG_PERSONAL_API_KEY` (a personal API key with the `person:write` scope), `POSTHOG_PROJECT_ID`, and `POSTHOG_APP_HOST` if the project is not in the EU cloud. Without them the PostHog step is skipped, which stops the erasure (below).
- **MailerLite**: `MAILERLITE_API_TOKEN`. Without it the MailerLite step is skipped, which stops the erasure (below).
- **Stripe**: `STRIPE_API_KEY`, for the subscriptions of orgs that go with the account.
- **`BAN_RECORD_HASH_KEY`**, the same value the REST API pods have: the ban record's hash is keyed with it, and sign-up (on the REST API pods) checks the hash with the same key. **Never rotate it**: the records kept under the old key stop matching, and those addresses can sign up again. Without it the ban record step fails for an account with a ban in force, and that erasure stops before the rows go; set it and run again. An account with no ban in force needs no record, and its erasure goes through without the key.
- It also reaches the scheduler and the notification manager over RPC, and Redis and RabbitMQ (the work stop), as the other services do.

A step is skipped only when its service is not configured on this stack at all. On a stack that uses the service, a skipped step **stops the erasure** before the account's rows go (they hold the address MailerLite needs and the expert IDs that name the memory graphs): the step is reported failed, the request stays open, the sweep logs an error, and the next run retries. Any error fails the step the same way. The request is never marked done while a service's data may be left. Fix the setting and run again. If the service can't be set up in time, delete the person's data there by hand, then run `erase-account` once with that service in `ACCOUNT_ERASURE_UNUSED_SERVICES`, for example `ACCOUNT_ERASURE_UNUSED_SERVICES='["posthog"]' poetry run cli erase-account <email> --apply`. The value is a JSON list: a bare `posthog` is rejected, and in `backend/.env` it would stop every service from starting. A stack that doesn't use a service at all, such as a self-hosted stack without PostHog, MailerLite or FalkorDB, lists it there for good: any of `buckets`, `falkordb`, `posthog`, `mailerlite`, `stripe`, or `postmark` for a stack that sends no email (the confirmation is then not sent). The confirmation to an address the mail provider won't deliver to (an earlier bounce or suppression) is not retried: the erasure goes on without it, since the address can never receive it. With no private bucket configured the files step sweeps the pod's local data directory and counts as `buckets` skipped, so a stack that keeps files on disk lists `buckets`. The files step sweeps every configured bucket, and also the pod's local data directory whenever the private bucket is unset, since workspace files are kept on disk then. The erasure's PostHog and MailerLite calls make at most three attempts on a 5xx, 429 or 408, then fail the step.

## Who runs the CLI, and where

The production operator, from a shell in a production **database manager** pod (`kubectl exec -it deploy/<database manager deployment> -n prod-agpt -- poetry run cli erase-account …`): the CLI needs the production database, Redis, RabbitMQ, the scheduler and notification services, FalkorDB, the buckets, Stripe, PostHog and MailerLite. The database manager is the pod that has the PostHog personal key and the MailerLite token, as the job runs there; other backend pods lack them, so the erasure would stop at those steps. `POSTHOG_APP_HOST` defaults to the EU cloud (`https://eu.posthog.com`); a project in the US cloud sets `https://us.posthog.com`. Each cloud refuses the other's keys, so a wrong host fails the step instead of finding no person.

In production, run it with care, as the pod serves every other service's database calls:

- **Check the pod's memory first**: `kubectl top pod -n prod-agpt` for the database manager pods, and pick one with room. One CLI run beside the idle service fits; a second process beside them got a dev pod killed for memory. A pod killed for memory takes the database manager down for every service using it, and kills the erasure mid-step (the rerun resumes it, above).
- **One command at a time**: wait for each to end before starting the next, and never run two in the same pod.
- **Save the terminal output of every production run**, dry runs included, where privacy requests are tracked: under `kubectl exec` the CLI's output, its "Account erasure of <user ID>: ..." log line included, reaches only your terminal, not Cloud Logging. It prints user IDs and counts, never the email.
- **"another run is erasing it" right after a crash** (the pod was killed, the terminal closed mid-run) is the dead run's lease: wait 5 minutes and run the same command again, which resumes the erasure.

The memory step is skipped only when `GRAPHITI_FALKORDB_HOST` is empty, which is how a stack without FalkorDB says so. Where it is set (production and dev set it in the shared config), a FalkorDB the pod cannot reach fails the step, and the erasure stops before the account's rows go; fix the connection and run it again. A self-hosted stack that does not run FalkorDB sets `GRAPHITI_FALKORDB_HOST=` (empty), since the default points at `localhost`.

Every command is a dry run unless it has `--apply`, and prints counts only, never the email. Every step is idempotent: a failed `--apply` stops with the account's rows still in place, and running the same command again resumes.

## A request sent by email

1. **Check the request.** Act only on a request sent from the account's own email address. One from any other address (even one naming the account) is answered by asking the person to write from that address, or to delete the account themselves in Settings › Account. Never act on a request about someone else's account.
2. **Check with the owner whether a refund is due, before closing.** Closing cancels the account's subscriptions at once, and nothing refunds anything. Whether, when and how much to refund is an open business decision; until it is made, ask the owner for each request. Make any refund in Stripe by hand.
3. **Close.** Pass the email or the user ID. Dry run first:

   ```sh
   poetry run cli erase-account sam@example.com --close
   poetry run cli erase-account sam@example.com --close --apply
   ```

   `--close` only closes the account, the way Settings does, with source `EMAIL`, and prints the date the 14 days end. The confirmation email is the one Settings sends, with its own reason line: "You asked us by email to delete your account." The owner can sign in and keep the account until that date, as from Settings. Closing also sends the anonymous `account_deletion_completed` count, with `surface: email`, which the tracking plan does not list yet. If Stripe refuses the plan cancel, nothing is closed: run it again. On an account already closed (from Settings, say) `--close` changes nothing and prints its date.

   A banned account may be closed too: deleting it is the owner's right, and the ban stays as it is. The erasure keeps a ban record for it (below).
   **When the owner asked to be erased at once**, skip the 14 days: `--close --now` closes the account if it is open (the confirmation email then says it is being erased, with no date to keep it) or, for one closed already, ends its 14 days now, and erases it in the same run, dry run first. It checks the blockers before it closes anything, so a blocked account is left open, not closed and stuck:

   ```sh
   poetry run cli erase-account sam@example.com --close --now
   poetry run cli erase-account sam@example.com --close --now --apply
   ```

   The owner can no longer keep the account after that. Then go on to step 5.
4. **Erase**, after the 14 days. The job does it within the hour. To run it by hand:

   ```sh
   poetry run cli erase-account sam@example.com
   poetry run cli erase-account sam@example.com --apply
   ```

   It refuses while a blocker remains, and nothing is deleted then:
   - `not_closed`: the account has no deletion request. Close it first (step 3).
   - `not_due`: the 14 days are not over; the blocker names the date. Wait, or let the job erase it.
   - `shared_orgs_without_another_owner`: the account is the only owner of an organization with other active members. Ask the requester to make someone else an owner, or decide with the org's members. Invited, suspended and removed members do not count: an organization only they are left in goes with the account, and its subscription is cancelled. The job logs this account at every run until it is resolved.
   - `oauth_apps_in_use`: the account registered an OAuth app (it is the app's developer) that another account still holds a live grant to. Erasing the account would delete the app and cut those accounts off, and the app is not handed to the "deleted user" account: the developer still holds its client secret and controls its redirect URIs, and nobody here could manage it. Decide with the product team, then either move the app to another owner (set `OAuthApplication.ownerId` to their user ID) or, if it should stop working, disable it (`isActive` = false). A disabled app, or one whose grants are all revoked or expired, goes with the account. The job logs this account at every run until it is resolved.
   - `invalid_account_id`: the ID is not a user ID. Nothing runs.
   - `system_account`: the "deleted user" placeholder (below) is never erased.

   It exits non-zero unless every step finished, and also when the owner kept the account between the dry run and the apply ("nothing ran: kept by its owner, ...").
5. **Confirm.** Run the dry run again with the user ID (the email no longer resolves) and check every count is 0. It reports the account as not found, which is expected now.
6. **Follow up by hand** on what the routine does not cover (below), then reply to the requester that their account and data are deleted. Record the request and its dates wherever privacy requests are tracked, by user ID; do not paste the email into tickets or channels.

To erase every closed account by hand (when the job is off, or to see what it would do):

```sh
poetry run cli erase-account --pending
poetry run cli erase-account --pending --apply
```

The apply erases the accounts the dry run listed as unblocked.

## What it does, in order

Before any step, the erasure claims the request (`startedAt`, above). The account's rows go last, because every earlier step finds its data through them; a step that fails stops the erasure there, and the next run (or a rerun of the CLI) resumes.

1. **Work** (`stop_work`): reruns closing's `stop_user_work`, checks that no schedule is left (it never raises, so a scheduler it couldn't reach would otherwise pass unnoticed: a schedule still listed fails the step), and removes the memory crons (nightly batch, community rebuild, skill learning) of the account and of each expert it owns. A schedule or trigger whose removal failed when the account was closed is not retried otherwise, and neither the scheduler nor the webhook ingress checks for a closed account. Closing leaves the memory crons registered so they resume if the account is kept (`is_user_refused` stops them doing anything meanwhile); once the account is erased nothing else would remove them. A scheduler that can't be reached fails the step.
2. **Confirmation**: if closing could not send the account-closed email (`confirmationSentAt` is null), sends it now, while the address still exists, and records it. By now the 14 days are over, so it says the account is being erased instead of naming a last day to keep it. A send that fails stops the erasure, so the next run tries again.
3. **Org subscriptions**: expires the open Checkouts and cancels the subscriptions of the organizations that go with the account, when their Stripe customer is not the account's own (closing cancelled that one).
4. **Shared agents**: hands the agents other accounts use to the "deleted user" account (below), with their marketplace listings.
5. **Marketplace**: its reviews, its other listings (which closing took down) with their versions and search embeddings, and every skill listing it published, with its versions and files. A skill listing is never handed over: an install copies the skill into the installer's own skills folder, so people who installed it keep their copy. Left to the account row's delete, the listing would stay with no owner, which marks a skill the platform wrote.
6. **Files**: everything under its prefixes in every bucket (private, public and legacy): `workspaces/{workspace id}/` (workspace files), `workspaces/cli-sessions/{user id}/` (AutoPilot transcripts), `uploads/users/{user id}/` (uploads, onboarding voice recordings), `users/{user id}/` (marketplace media and avatars, including their public copies) and `oauth-apps/{app id}/` for OAuth apps it owns. A self-hosted stack's folders on disk likewise. Media a listing that stays still shows is kept (below).
7. **Memory**: AutoPilot's memory graphs in FalkorDB, Otto's and one per expert it owns. Skipped on a stack with no FalkorDB host configured (above).
8. **PostHog**: the person, with its events and recordings (PostHog deletes events in a weekly batch).
9. **MailerLite**: "forgets" the subscriber, which removes them now and erases their data within 30 days.
10. **Ban record**: if the auth identity is still banned (a ban with no expiry, or one that has not run out), keeps a ban record (below) before the identity goes.
11. **Database**: first the account's own runs, in batches. Then, in one transaction: strips the personal content from its credit ledger (kept, below); deletes its refund requests, private search embeddings, its chats (deleted here rather than left to the account's cascade, since a database can lack that foreign key: dev did on 11 Nov 2026; messages, shared files and links go with them), node key-value data, stored credentials, pending invitations to it, its webhook rows, the organizations only it belongs to (personal org included, with their ledger, teams, credentials and audit log), its presets with their saved inputs, the events of its notification batches (both would otherwise stay behind with their link set to null), its auth identity with sessions and sign-in accounts, and the account row, which cascades to everything else it owns: its other agents, library, experts, workspace records, API keys, OAuth apps (none another account still uses: that blocks the erasure, above) and tokens, notification batches and preferences, onboarding and the rest. The identity goes in the same transaction as the account, so the orphan-healing job (`heal_orphaned_auth_identities`) never sees an identity without an account and re-creates it.
12. **Request**: sets `completedAt` on the request this erasure claimed (by its ID, with `startedAt` set), never on another row of the same user ID. The row itself stays, with the user ID, so backend auth keeps refusing a token issued before the account was closed, and nothing re-creates the account from it.

Chats and runs the account created inside a shared organization belong to the account and go with it. Its agents there stay, since the org's members can open them (below).

## Agents other accounts use

Deleting an agent another account uses would take that account's library entry, presets and runs with it. So instead of deleting it, the erasure hands it to a placeholder account and the agent keeps working for everyone using it: their library entries, presets, schedules, triggers and runs stay as they are.

An agent counts as used by others when another account has it in its library or presets, has run it, lists it in the marketplace, or an expert of theirs (or a roster template) installed its listing. Its own listing alone does not count: closing the account took every listing of its own down, so nobody can add the agent any more, and one nobody has added is deleted. So does an agent at the home of an organization someone else is an active member of, since every member can open it without adding it anywhere, and a team agent while another active member of that team can open it; it keeps its org and team. A team agent only the account could open goes with the account. Every version of such an agent goes over, its unpublished versions included, since library entries and runs pin versions. So does every agent it runs as a sub-agent, at any depth, because a sub-agent only loads when it has the same owner as the agent at the top.

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
- **Ban records** (`AccountBanRecord`), for an account erased while banned: a keyed hash of its email (HMAC-SHA256 keyed with `BAN_RECORD_HASH_KEY`, a key of its own, so the table alone does not say whose addresses it holds and rotating `ENCRYPTION_KEY` leaves the records matching), the date, and the ban's category (`ADMIN`, the only way an account is banned today; the admin's free-text reason is never copied, as it may name the person). A record lasts as long as the ban would have, and **3 years** at most (`deleteAfter`); each job run purges the ones past their date. A rerun keeps the record the first run wrote, so retries never push its end out. With no `BAN_RECORD_HASH_KEY` the ban record step fails (the ban would be lost with the identity), so the erasure waits for the key; sign-up without it finds no record and lets the address in. **Never rotate `BAN_RECORD_HASH_KEY`**: the records kept under the old key would stop matching, and nothing carries them over. While a record lasts, sign-up with that address is refused: Better Auth's sign-up checks (`frontend/src/lib/auth/ban-record.ts`, for email sign-up and a first social sign-in) ask the backend (`POST /api/auth/email/ban-record`, frontend service token) and refuse a match with the generic "Sign-up isn't available" message. The lookup fails open: a backend that is down never blocks a sign-up. Only sign-up is checked: an admin creating the account from /admin/users is not, and neither is an existing account changing its email to that address.
- **Trial claims**: `SubscriptionTrialClaim` keeps hashes that stop the same person claiming a second free trial with a new account. They carry no account ID, on purpose.
- **Our cost records**: `PlatformCostLog` rows stay with their user ID set to null.
- **Audit log rows** of shared organizations the account acted in stay, with its user ID.
- **What it created in shared organizations**: team and org credentials it added stay with the team or org, and rows it initiated there (invitations it sent, transfers, org ledger entries, teams) keep its user ID.
- **Known gap, on purpose for now**: when a member of an organization that still has other members is erased, the org ledger rows they started (`OrgCreditTransaction`, its `initiatedByUserId` and `metadata`) and the org's shared listing media are left as they are. Organizations are frozen and hardly used, so this waits for SECRT-3365.
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
- **Changing a ban record**: to let an address sign up again before its record runs out, delete its row from `AccountBanRecord`. Find it by computing the hash with `backend.data.account_erasure.ban_record.email_hash` in a backend shell; the table can't be searched by email.
