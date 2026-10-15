# Deleting an account (privacy requests)

How to handle a request to delete an account, such as one sent to privacy@agpt.co, with the `delete-account` CLI. The CLI and the self-service deletion in settings (SECRT-2877) run the same routine, `backend/data/account_deletion/routine.py` (SECRT-2854).

Deletion has two steps, 14 days apart. The **lock** stops the account straight away: nobody can sign in, its subscriptions are cancelled, its schedules and runs stop. The **deletion** runs once 14 days have passed since the lock, so a request made by mistake can still be undone with an **unlock**.

We promise to delete within 30 days of the request. Lock within a week of it: the deletion is then due 14 days later, which leaves time to sort out a failed run or a blocker.

## Who runs it, and where

The production operator, from a shell in a production backend pod: the CLI needs the production database, Redis, RabbitMQ, the scheduler service, FalkorDB, the buckets, Stripe, PostHog and MailerLite, which only those pods reach. Merging this does not configure anything: PostHog deletion needs `POSTHOG_PERSONAL_API_KEY` (a personal API key with the `person:write` scope) and `POSTHOG_PROJECT_ID` in the backend's secrets. Without them the PostHog step is skipped and says so.

Every command is a dry run unless it has `--apply`, and prints counts only, never the email. Every step is idempotent: a failed `--apply` stops with the account's rows still in place, and running the same command again resumes.

## Steps

1. **Check the request.** It must come from the account's own email address, or prove control of it (reply from that address). Never act on a request about someone else's account.
2. **Check with the owner whether a refund is due, before cancelling anything.** The lock cancels the account's subscriptions at once, and nothing in the routine refunds anything. Whether, when and how much to refund is an open business decision; until it is made, ask the owner for each request. Make any refund in Stripe by hand.
3. **Lock.** Pass the email or the user ID. Run it without `--apply` first to see what it would do:

   ```sh
   poetry run cli delete-account sam@example.com --lock
   poetry run cli delete-account sam@example.com --lock --apply
   ```

   It bans sign-in and ends the account's sessions, opts it out of marketing, deletes its schedules, stops its runs and cancels its Stripe subscriptions. It deletes no data. It prints the user ID and the date the deletion may run from; note both with the request. Locking again keeps the first date.
4. **If the requester changes their mind** before the deletion has run:

   ```sh
   poetry run cli delete-account <user-id> --unlock --apply
   ```

   It lifts the ban and the marketing opt-out the lock set (an opt-out the account had chosen itself stays) and clears the lock. Cancelled subscriptions stay cancelled and deleted schedules are not restored: tell the requester they need to subscribe again and set their schedules up again.
5. **Delete, once the 14 days have passed.** Dry run, then apply:

   ```sh
   poetry run cli delete-account <user-id>
   poetry run cli delete-account <user-id> --apply
   ```

   It refuses while a blocker remains, and nothing is deleted then:
   - `undo_window_open`: the lock is less than 14 days old; the count is the days left and the note says from when. `not_locked`: the account was never locked (or was unlocked). `--force` skips both, for example when a lock was made before this check existed, or the requester asks for it to go now in writing.
   - `shared_orgs_without_another_owner`: the account is the only owner of an organization with other members. Ask the requester to make someone else an owner, or decide with the org's members.

   It exits non-zero unless every step finished.
6. **Confirm.** Run the dry run with the user ID (the email no longer resolves) and check every count is 0. Run it once more an hour later: an access token issued before the lock stays valid for up to an hour, and a request made with it can re-create an empty account row. If one appears, apply again with `--force` (the new row has no lock).
7. **Follow up by hand** on what the routine does not cover (below), then reply to the requester that their account and data are deleted. Record the request and the dates of the lock and the deletion wherever privacy requests are tracked, by user ID; do not paste the email into tickets or channels.

## What it does, in order

The lock runs steps 1 to 4. The deletion runs them again (they are no-ops by then), then the rest. The account's rows go last, because every earlier step finds its data through them.

1. **Sign-in**: bans the auth identity and ends its sessions; opts the account out of marketing, so a MailerLite update queued meanwhile is dropped.
2. **Schedules**: agent and AutoPilot schedules, and the per-user memory, nightly and briefing crons.
3. **Runs**: stops the account's agent runs that are queued or running.
4. **Subscriptions**: cancels active and trialing Stripe subscriptions of the account and its own orgs, immediately. (The lock then records its date.)
5. **Integrations**: deregisters its webhooks at the providers and deprovisions managed credentials (AgentMail).
6. **Shared agents**: hands the agents other accounts use to the "deleted user" account (below), with their marketplace listings.
7. **Marketplace**: its reviews, and its other listings with their versions and search embeddings.
8. **Files**: everything under its prefixes in every bucket (private, public and legacy): `workspaces/{workspace id}/` (workspace files), `workspaces/cli-sessions/{user id}/` (AutoPilot transcripts), `uploads/users/{user id}/` (uploads, onboarding voice recordings), `users/{user id}/` (marketplace media and avatars, including their public copies) and `oauth-apps/{app id}/` for OAuth apps it owns. A self-hosted stack's folders on disk likewise. Media a listing that stays still shows is kept (below).
9. **Memory**: AutoPilot's memory graphs in FalkorDB, Otto's and one per expert it owns.
10. **PostHog**: the person, with its events and recordings (PostHog deletes events in a weekly batch).
11. **MailerLite**: "forgets" the subscriber, which removes them now and erases their data within 30 days.
12. **Database**: first the account's own runs, in batches. Then, in one transaction: strips the personal content from its credit ledger (kept, below); deletes its refund requests, private search embeddings, node key-value data, stored credentials, pending invitations to it, its webhook rows, the organizations only it belongs to (personal org included, with their ledger, teams, credentials and audit log), its auth identity with sessions and sign-in accounts, and the account row, which cascades to everything else it owns: its other agents, chats, library, experts, workspace records, API keys, OAuth apps and tokens, notifications, onboarding and the rest. The identity goes in the same transaction as the account, so the orphan-healing job (`heal_orphaned_auth_identities`) never sees an identity without an account and re-creates it.

Agents, chats and runs the account created inside a shared organization belong to the account and go with it, unless another account uses the agent.

## Agents other accounts use

Deleting an agent another account uses would take that account's library entry, presets and runs with it. So instead of deleting it, the deletion hands it to a placeholder account and the agent keeps working for everyone using it: their library entries, presets, schedules, triggers and runs stay as they are.

An agent counts as used by others when another account has it in its library or presets, has run it, lists it in the marketplace, or an expert of theirs (or a roster template) installed its listing; and when it has a live marketplace listing, even if nobody has added it yet. Every version of such an agent goes over, its unpublished versions included, since library entries and runs pin versions. So does every agent it runs as a sub-agent, because a sub-agent must have the same owner as its parent.

The placeholder is a platform account with ID `d83236d9-19c5-455f-90c8-4cc6d76d4dac`, email `deleted-user@agpt.invalid` and a marketplace profile named "Deleted user" (handle `deleted-user`). It has no auth identity, so nobody can sign in as it, and every email and marketing setting is off. The first deletion that needs it creates it; the routine refuses to lock or delete it.

What changes for the agent's listing:
- It stays live, shown as by "Deleted user". The creator's name, avatar, bio and links go with their profile. Its URL changes, since it carries the creator's handle; a slug the placeholder already uses gets a number added.
- Its images and videos stay where they are, under the old account's `users/{user id}/` path, which the files step leaves alone. That applies to every listing that stays, and to the images other accounts' library entries copied from them.
- A version still waiting for review is rejected, so nothing the creator had not published gets published after they asked to be deleted.
- Reviews other accounts left stay; the creator's own reviews of other agents are deleted.
- Other accounts' library entries name "Deleted user" as the creator.

The agent's content stays as the creator last saved it, node inputs included. Credentials it referenced are deleted with the account, so they work for nobody; each account running the agent uses its own.

## What it keeps, and why

- **The credit ledger** (`CreditTransaction`): billing records are kept for six years ("Billing and payment records: 6 years" in the privacy policy, as UK tax law requires). The rows keep their user ID, amounts, types, dates, running balances and Stripe IDs. Their metadata is cut down to IDs and amounts: the block inputs, free-text reasons, and the names, emails and addresses in stored copies of Stripe objects go (`backend/data/account_deletion/ledger.py`). The admin credit views show these rows without an email.
- **Stripe billing records**: the customer, invoices, payments and refunds stay in Stripe; the customer's metadata still holds the old user ID.
- **Agents other accounts use**, their listings and those listings' media (above).
- **Trial claims**: `SubscriptionTrialClaim` keeps hashes that stop the same person claiming a second free trial with a new account. They carry no account ID, on purpose.
- **Our cost records**: `PlatformCostLog` rows stay with their user ID set to null.
- **Audit log rows** of shared organizations the account acted in stay, with its user ID.
- **Langfuse traces** expire on their own after 90 days.
- **Database backups** age out on their retention schedule.

## Not covered: do by hand, or let expire

- **Refunds**: see step 2. An open business decision.
- **The org ledger** (`OrgCreditTransaction`) of an organization only the account belongs to is still deleted with the org. Personal orgs bill the account's own ledger, so this only matters for a team org with one member left; whether the six-year rule applies to it as well is not decided.
- **API keys and platform OAuth tokens keep working while the account is locked.** The lock bans sign-in only; they are deleted with the account at the end of the window. Anything they start meanwhile is stopped again by the deletion.
- **OAuth grants at the providers** (Google, GitHub, ...). The routine deletes the tokens with the account but does not revoke them; the self-service endpoint does, through the credential-deletion route's revocation. For an email request, ask the requester to revoke the app at the provider if it matters to them.
- **Skill listings** the account published in the marketplace are deleted with its profile, not handed over.
- **E2B volumes** named `autogpt-user-{user id}` and `autogpt-expert-{expert id}` (desktop blocks). Sandboxes stop on their own timeout; volumes do not. Delete them in the E2B dashboard.
- **Mem0** memories stored with the user ID by the Mem0 blocks: delete with Mem0's `delete_all(user_id=...)`.
- **Linear** customer created for feature requests (`externalId` = user ID, named with the email).
- **Ayrshare** profile titled `User {user id}-...`: Ayrshare has no API to delete it.
- **LaunchDarkly** context and **Sentry** user (ID only): no deletion API used here.
- **Redis keys** with the user ID (rate limits, usage counters, caches, AutoPilot session state) expire on their own TTLs, the longest after 90 days.
- **AutoPilot turns in flight** are not stopped; their chats are deleted, so a turn that finishes afterwards has nowhere to write.

## For the self-service endpoint (SECRT-2877)

From `backend/data/account_deletion/routine.py`:

- On the request, call `lock_account(user_id, dry_run=False)`. It stops the account and records the date.
- To undo within the window, call `unlock_account(user_id, dry_run=False)`.
- From 14 days after the lock (a scheduled job, say), call `delete_account(user_id, dry_run=False)`. It raises `AccountDeletionBlocked` (with the report) before deleting anything while the window is open or another blocker needs the user's action, and returns a `DeletionReport`; `report.complete` is true only when every step finished. A second call on a deleted account finds nothing and completes, so the job can retry safely. It can run for minutes on a large account, so run it in the background rather than in a request.

The routine repeats the stop steps the endpoint may already have run (sign-in, schedules, runs, subscriptions); they are no-ops by then.
