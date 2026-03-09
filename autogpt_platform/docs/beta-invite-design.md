# Beta Invite Pre-Provisioning Design

## Problem

The current signup path is split across three places:

- Supabase creates the auth user and password.
- The backend lazily creates `platform.User` on first authenticated request in [`backend/backend/data/user.py`](/Users/swifty/work/agpt/AutoGPT/autogpt_platform/backend/backend/data/user.py).
- A Postgres trigger creates `platform.Profile` after `auth.users` insert in [`backend/migrations/20250205100104_add_profile_trigger/migration.sql`](/Users/swifty/work/agpt/AutoGPT/autogpt_platform/backend/migrations/20250205100104_add_profile_trigger/migration.sql).

That works for open signup plus a DB-level allowlist, but it does not give the platform a durable object representing:

- a beta invite before the person signs up
- pre-computed onboarding data tied to an email before auth exists
- the distinction between "invited", "claimed", "active", and "revoked"

It also makes first-login setup racey because `User`, `Profile`, `UserOnboarding`, and `CoPilotUnderstanding` are not created from one source of truth.

## Goals

- Allow staff to invite a user before they have a Supabase account.
- Pre-create the platform-side user record and related data before first login.
- Keep Supabase responsible for password entry, email verification, sessions, and OAuth providers.
- Support both password signup and magic-link / OAuth signup for invited users.
- Preserve the existing ability to populate Tally-derived understanding and onboarding defaults.
- Make first login idempotent and safe if the user retries or signs in with a different method.

## Non-goals

- Replacing Supabase Auth.
- Building a full enterprise identity management system.
- Solving general-purpose team/org invites in this change.

## Proposed model

Introduce invite-backed pre-provisioning with email as the pre-auth identity key, then bind the invite to the Supabase `auth.users.id` when the user claims it.

### New tables

#### `BetaInvite`

Represents an invitation sent to a person before they create credentials.

Suggested fields:

- `id`
- `email` unique
- `status` enum: `PENDING`, `CLAIMED`, `EXPIRED`, `REVOKED`
- `inviteTokenHash` nullable
- `invitedByUserId` nullable
- `expiresAt` nullable
- `claimedAt` nullable
- `claimedAuthUserId` nullable unique
- `metadata` jsonb
- `createdAt`
- `updatedAt`

`metadata` should hold operational fields only, for example:

- source: manual import, admin UI, CSV
- cohort: beta wave name
- notes
- original tally email if normalization differs

#### `PreProvisionedUser`

Represents the platform-side user state that exists before Supabase auth is bound.

Suggested fields:

- `id`
- `inviteId` unique
- `email` unique
- `authUserId` nullable unique
- `status` enum: `PENDING_CLAIM`, `ACTIVE`, `MERGE_REQUIRED`, `DISABLED`
- `name` nullable
- `timezone` default `not-set`
- `emailVerified` default `false`
- `metadata` jsonb
- `createdAt`
- `updatedAt`

This is the durable record staff can enrich before login.

#### `PreProvisionedUserSeed`

Stores structured seed data that should become first-class user records on claim.

Suggested fields:

- `id`
- `preProvisionedUserId` unique
- `profile` jsonb nullable
- `onboarding` jsonb nullable
- `businessUnderstanding` jsonb nullable
- `promptContext` jsonb nullable
- `createdAt`
- `updatedAt`

This avoids polluting the main `User.metadata` blob and gives explicit ownership over what is seed data versus ongoing user-authored data.

## Why not pre-create `platform.User` directly?

`platform.User.id` is currently designed to equal the Supabase user id in [`backend/schema.prisma`](/Users/swifty/work/agpt/AutoGPT/autogpt_platform/backend/schema.prisma). Pre-creating `User` with a fake UUID would ripple through every FK and complicate the eventual bind. Reusing email as the join key inside `User` is also not safe enough because the rest of the system assumes `User.id` is the canonical identity.

A separate pre-provisioning layer is lower risk:

- it avoids breaking every relation off `User`
- it keeps the existing auth token model intact
- it gives a clean migration path to merge into `User` once a Supabase identity exists

## Claim flow

### 1. Staff creates invite

Admin API or script:

1. Create `BetaInvite`.
2. Create `PreProvisionedUser`.
3. Create `PreProvisionedUserSeed`.
4. Optionally fetch and store Tally-derived understanding immediately by email.
5. Optionally send invite email containing a claim link.

### 2. User opens signup page

Two supported modes:

- direct signup with invited email
- invite-link signup with a short-lived token

Preferred UX:

- `/signup?invite=<token>`
- frontend resolves token to masked email and invite status
- email field is prefilled and locked by default

### 3. User creates Supabase account

Frontend still calls Supabase `signUp`, but include invite context in user metadata:

- `invite_id`
- `invite_email`

This is useful for debugging but should not be the source of truth.

### 4. First authenticated backend call activates invite

Replace the current pure `get_or_create_user` behavior with:

1. Read JWT `sub` and `email`.
2. Look for existing `platform.User` by `id`.
3. If found, return it.
4. Else look for `PreProvisionedUser` by normalized email and `status = PENDING_CLAIM`.
5. In one transaction:
   - create `platform.User` with `id = auth sub`
   - bind `PreProvisionedUser.authUserId = auth sub`
   - mark `PreProvisionedUser.status = ACTIVE`
   - mark `BetaInvite.status = CLAIMED`
   - create or upsert `Profile`
   - create or upsert `UserOnboarding`
   - create or upsert `CoPilotUnderstanding`
   - create `UserWorkspace` if required for the product experience
6. If no pre-provisioned record exists, either:
   - reject access for closed beta, or
   - fall back to normal creation if a feature flag allows open signup

This activation logic should live in the backend service, not a Postgres trigger, because it needs to coordinate across multiple platform tables and apply merge rules.

## Data merge rules

When activating a pre-provisioned invite:

- `User`
  - source of truth for `id` is Supabase JWT `sub`
  - source of truth for `email` is Supabase auth email
  - `name` prefers pre-provisioned value, falls back to auth metadata name

- `Profile`
  - if seed profile exists, use it
  - else create the current generated default
  - never overwrite an existing profile if activation is retried

- `UserOnboarding`
  - seed values are initial defaults only
  - use `upsert` with create-on-missing semantics

- `CoPilotUnderstanding`
  - seed from stored Tally extraction if present
  - skip Tally backfill on first login if this already exists

- prompt/tally-specific context
  - do not jam this into `User.metadata` unless no better typed model exists
  - use `PreProvisionedUserSeed.promptContext` now, migrate to typed tables later if product solidifies

## Invite enforcement

The current closed-beta gate appears to rely on a Supabase-side DB error surfaced as "not allowed" in [`frontend/src/app/api/auth/utils.ts`](/Users/swifty/work/agpt/AutoGPT/autogpt_platform/frontend/src/app/api/auth/utils.ts).

That should evolve to:

### Short term

Keep the current Supabase signup restriction, but have it check `BetaInvite`/`PreProvisionedUser` instead of a raw allowlist.

### Medium term

Stop using a generic DB exception as the main product signal and expose a backend endpoint:

- `POST /internal/beta-invites/validate`
- `GET /internal/beta-invites/:token`

Then the frontend can fail earlier with a specific state:

- invited and ready
- invite expired
- already claimed
- not invited

Supabase should still remain the hard gate for credential creation, but the UI should stop depending on opaque trigger failures for normal control flow.

## Trigger changes

The `auth.users` trigger that creates `platform.User` and `platform.Profile` should be removed once activation logic is live.

Reason:

- it cannot see or safely apply pre-provisioned seed data
- it only handles create, not merge/bind
- it duplicates responsibility already present in `get_or_create_user`

Interim state during rollout:

- keep trigger disabled for production once backend activation ships
- keep a backfill script for any auth users created during the transition

## API surface

### Admin APIs

- `POST /admin/beta-invites`
  - create invite + pre-provisioned user + seed data
- `POST /admin/beta-invites/bulk`
  - CSV import for beta cohorts
- `POST /admin/beta-invites/:id/resend`
- `POST /admin/beta-invites/:id/revoke`
- `GET /admin/beta-invites`

### Public/auth-adjacent APIs

- `GET /beta-invites/lookup?token=...`
  - return safe invite info for signup page
- `POST /beta-invites/claim-preview`
  - optional: check whether an email is invited before attempting Supabase signup

### Internal service function changes

- Replace `get_or_create_user` with `get_or_activate_user`
- keep a compatibility wrapper if needed to limit churn

## Suggested backend implementation

### New module

- `backend/backend/data/beta_invite.py`

Responsibilities:

- create invite
- resolve invite token
- normalize email
- activate pre-provisioned user
- revoke / expire invite

### Existing module changes

- [`backend/backend/data/user.py`](/Users/swifty/work/agpt/AutoGPT/autogpt_platform/backend/backend/data/user.py)
  - move lazy creation logic into activation-aware flow

- [`backend/backend/api/features/v1.py`](/Users/swifty/work/agpt/AutoGPT/autogpt_platform/backend/backend/api/features/v1.py)
  - `/auth/user` should call activation-aware function
  - only run background Tally population if no seeded understanding exists

- [`backend/backend/data/tally.py`](/Users/swifty/work/agpt/AutoGPT/autogpt_platform/backend/backend/data/tally.py)
  - add a reusable "seed from email" function for invite creation time

## Suggested frontend implementation

### Signup

- read invite token from URL
- call invite lookup endpoint
- prefill locked email when token is valid
- if invite is invalid, show specific error, not generic waitlist modal

Files likely affected:

- [`frontend/src/app/(platform)/signup/page.tsx`](/Users/swifty/work/agpt/AutoGPT/autogpt_platform/frontend/src/app/(platform)/signup/page.tsx)
- [`frontend/src/app/(platform)/signup/actions.ts`](/Users/swifty/work/agpt/AutoGPT/autogpt_platform/frontend/src/app/(platform)/signup/actions.ts)
- [`frontend/src/components/auth/WaitlistErrorContent.tsx`](/Users/swifty/work/agpt/AutoGPT/autogpt_platform/frontend/src/components/auth/WaitlistErrorContent.tsx)

### Admin UI

Add a simple internal page for beta invites instead of manually editing allowlists.

Possible location:

- `frontend/src/app/(platform)/admin/users/invites/page.tsx`

## Rollout plan

### Phase 1: schema and service layer

- add `BetaInvite`, `PreProvisionedUser`, `PreProvisionedUserSeed`
- implement activation transaction
- keep existing trigger/allowlist in place

### Phase 2: admin creation path

- add admin API or CLI script
- support single invite and CSV bulk upload
- seed Tally/business understanding during invite creation

### Phase 3: signup UX

- invite token lookup
- better invite state messaging
- preserve existing closed beta modal for non-invited traffic

### Phase 4: remove legacy coupling

- disable `auth.users` profile trigger
- simplify `get_or_create_user`
- migrate allowlist logic to invite tables

## Edge cases

### Existing auth user, no platform user

This already happens today. Activation flow should treat it as:

- if auth email matches a pending pre-provisioned invite, bind and activate
- else create a plain `User` only if open-signup feature flag is enabled

### Existing platform user, invited again

Do not create another `PreProvisionedUser`. Create a second invite only if product explicitly wants re-invites. Otherwise reject as duplicate.

### Email changed after invite

Support admin-side reissue:

- revoke old invite
- create new invite with new email
- move seed data forward

Do not automatically bind across unrelated email addresses.

### OAuth signup

OAuth provider signups still work as long as the resulting Supabase email matches the invited email. If the provider returns a different email, activation should fail with a clear UI message.

### Tally data arrives after invite creation

Allow re-seeding before claim if `CoPilotUnderstanding` has not yet been created on activation.

## Migration notes

### Existing beta users

No immediate migration required for already-active users. This system is mainly for future invited users.

### Existing allowlist entries

Backfill them into `BetaInvite` plus `PreProvisionedUser`, then swap the Supabase gating logic to consult invite tables instead of the legacy allowlist table/trigger.

## Recommendation

Implement the pre-provisioning layer first and keep `platform.User` bound to Supabase `auth.users.id`. That is the lowest-risk design because it respects the existing identity model while giving the business exactly what it needs:

- invite someone before signup
- compute and store Tally/onboarding/prompt defaults before first login
- activate those defaults atomically when the user actually creates credentials

## First implementation slice

The smallest useful slice is:

1. Add `BetaInvite`, `PreProvisionedUser`, and `PreProvisionedUserSeed`.
2. Add a backend admin endpoint to create an invite from email plus optional seed payload.
3. Change `/auth/user` activation logic to bind and materialize seeded `Profile`, `UserOnboarding`, and `CoPilotUnderstanding`.
4. Keep the existing signup UI, but validate invite membership before or during signup.

That delivers the core behavior without needing the full admin UI on day one.
