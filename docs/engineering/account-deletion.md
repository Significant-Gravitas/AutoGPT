# Deleting an account (privacy requests)

How to handle a request to delete an account, such as one sent to privacy@agpt.co, with the `delete-account` CLI. The CLI and the self-service deletion in settings (SECRT-2877) run the same routine, `backend/data/account_deletion/routine.py` (SECRT-2854).

We promise to delete within 30 days of the request. Start within a week, so a failed run or a blocker has time to be sorted out.

## Who runs it, and where

The production operator, from a shell in a production backend pod: the CLI needs the production database, Redis, RabbitMQ, the scheduler service, FalkorDB, the buckets, Stripe, PostHog and MailerLite, which only those pods reach. Merging this does not configure anything: PostHog deletion needs `POSTHOG_PERSONAL_API_KEY` (a personal API key with the `person:write` scope) and `POSTHOG_PROJECT_ID` in the backend's secrets. Without them the PostHog step is skipped and says so.

## Steps

1. **Check the request.** It must come from the account's own email address, or prove control of it (reply from that address). Never act on a request about someone else's account.
2. **Dry run.** Pass the email or the user ID:

   ```sh
   poetry run cli delete-account sam@example.com
   ```

   It prints the user ID, any blockers, and per step what would go, as counts. It never prints the email. Nothing is changed.
3. **Resolve blockers**, if any. Nothing is deleted while one remains:
   - `shared_orgs_without_another_owner`: the account is the only owner of an organization with other members. Ask the requester to make someone else an owner, or decide with the org's members.
   - `agents_used_by_other_accounts`: other accounts have this account's agents in their library or presets. Removing them takes the agent out of those libraries, and their runs of it go too. Once that is agreed (product owner), pass `--remove-shared-agents`.
   - `agents_listed_by_other_accounts`: another account's marketplace listing uses one of these agents. Remove that listing by hand first.
4. **Apply.** The CLI runs the dry run again, asks for confirmation, then deletes:

   ```sh
   poetry run cli delete-account <user-id> --apply
   ```

   It exits non-zero unless every step finished. A failed step stops the run with the account's rows still in place, so running the same command again resumes; every step is idempotent.
5. **Confirm.** Run the dry run with the user ID (the email no longer resolves) and check every count is 0. Run it once more an hour later: an access token issued before the deletion stays valid for up to an hour, and a request made with it can re-create an empty account row. If one appears, apply again.
6. **Follow up by hand** on what the routine does not cover (below), then reply to the requester that their account and data are deleted. Record the request and the date it was completed wherever privacy requests are tracked, by user ID; do not paste the email into tickets or channels.

## What it deletes, in order

Each step is safe to repeat. The account's rows go last, because every earlier step finds its data through them.

1. **Sign-in**: bans the auth identity and ends its sessions; opts the account out of marketing, so a MailerLite update queued meanwhile is dropped.
2. **Schedules**: agent and AutoPilot schedules, and the per-user memory, nightly and briefing crons.
3. **Runs**: stops agent runs that are queued or running.
4. **Subscriptions**: cancels active and trialing Stripe subscriptions of the account and its own orgs, immediately.
5. **Integrations**: deregisters its webhooks at the providers and deprovisions managed credentials (AgentMail).
6. **Marketplace**: its reviews, its listings with their versions and search embeddings.
7. **Files**: everything under its prefixes in every bucket (private, public and legacy): `workspaces/{workspace id}/` (workspace files), `workspaces/cli-sessions/{user id}/` (AutoPilot transcripts), `uploads/users/{user id}/` (uploads, onboarding voice recordings), `users/{user id}/` (marketplace media and avatars, including their public copies) and `oauth-apps/{app id}/` for OAuth apps it owns. A self-hosted stack's folders on disk likewise.
8. **Memory**: AutoPilot's memory graphs in FalkorDB, Otto's and one per expert it owns.
9. **PostHog**: the person, with its events and recordings (PostHog deletes events in a weekly batch).
10. **MailerLite**: "forgets" the subscriber, which removes them now and erases their data within 30 days.
11. **Database**, in one transaction: the account's credit ledger and refund requests, private search embeddings, node key-value data, stored credentials, pending invitations to it, its webhook rows, the organizations only it belongs to (personal org included, with their ledger, teams, credentials and audit log), its auth identity with sessions and sign-in accounts, and the account row, which cascades to everything else it owns: agents, runs, chats, library, experts, workspace records, API keys, OAuth apps and tokens, notifications, onboarding and the rest. The identity goes in the same transaction as the account, so the orphan-healing job (`heal_orphaned_auth_identities`) never sees an identity without an account and re-creates it.

Agents, chats and runs the account created inside a shared organization belong to the account and go with it. Media it uploaded for an organization's listing goes too.

## What it keeps, and why

- **Stripe billing records**: the customer, invoices, payments and refunds stay in Stripe. They are the records accounting and tax law require; the customer's metadata still holds the old user ID.
- **The internal credit ledger is not kept.** `CreditTransaction` was meant to outlive the account (#10917), but its foreign key is `ON DELETE NO ACTION`, which refuses to delete an account that has any row. The routine deletes the account's rows; Stripe holds the money side. Keeping them would take a migration that drops the foreign key and its Prisma relation, which the admin credit views query through.
- **Trial claims**: `SubscriptionTrialClaim` keeps hashes that stop the same person claiming a second free trial with a new account. They carry no account ID, on purpose.
- **Our cost records**: `PlatformCostLog` rows stay with their user ID set to null.
- **Audit log rows** of shared organizations the account acted in stay, with its user ID.
- **Langfuse traces** expire on their own after 90 days.
- **Database backups** age out on their retention schedule.

## Not covered: do by hand, or let expire

- **OAuth grants at the providers** (Google, GitHub, ...). The routine deletes the tokens with the account but does not revoke them; the self-service endpoint does, through the credential-deletion route's revocation. For an email request, ask the requester to revoke the app at the provider if it matters to them.
- **E2B volumes** named `autogpt-user-{user id}` and `autogpt-expert-{expert id}` (desktop blocks). Sandboxes stop on their own timeout; volumes do not. Delete them in the E2B dashboard.
- **Mem0** memories stored with the user ID by the Mem0 blocks: delete with Mem0's `delete_all(user_id=...)`.
- **Linear** customer created for feature requests (`externalId` = user ID, named with the email).
- **Ayrshare** profile titled `User {user id}-...`: Ayrshare has no API to delete it.
- **LaunchDarkly** context and **Sentry** user (ID only): no deletion API used here.
- **Redis keys** with the user ID (rate limits, usage counters, caches, AutoPilot session state) expire on their own TTLs, the longest after 90 days.
- **AutoPilot turns in flight** are not stopped; their chats are deleted, so a turn that finishes afterwards has nowhere to write.

## For the self-service endpoint (SECRT-2877)

Call `delete_account(user_id, dry_run=False)` from `backend/data/account_deletion/routine.py`. It raises `AccountDeletionBlocked` (with the report) before deleting anything when a blocker needs the user's action, and returns a `DeletionReport`; `report.complete` is true only when every step finished. A second call on a deleted account finds nothing and completes, so the endpoint can return 200 again. The routine repeats the stop steps the endpoint runs first (sign-in, schedules, runs, subscriptions), which are no-ops by then. It can run for minutes on a large account, so run it in the background rather than in the request.
