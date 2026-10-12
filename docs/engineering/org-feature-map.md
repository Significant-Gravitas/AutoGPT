# Organization Feature Map

A capability map for org/team tenancy on the AutoGPT Platform: what the
foundation shipped in the org-workspace PR series provides, and the feature
surface we expect to build on top of it. Each section lists the existing
hooks in the codebase that a feature would attach to, so scoping starts from
real anchor points instead of a blank page.

## Foundation (shipped)

- **Schema**: `Organization`, `Team`, `OrgMember`, `TeamMember`,
  `OrgInvitation`, `TeamInvite`, `OrganizationProfile`, `OrganizationAlias`,
  `OrgBalance`, `OrgCreditTransaction`, `OrganizationSubscription`,
  `OrganizationSeatAssignment`, `TransferRequest`, `AuditLog`
- **Request context**: `RequestContext` (`org_id`, `team_id`, role flags,
  `seat_status`) resolved per-request via `get_request_context`, with
  `X-Org-Id` / `X-Team-Id` header overrides and personal-org fallback
- **Execution context**: `ExecutionContext.organization_id` / `team_id`
  threaded through graph executions, webhooks, schedules, copilot runs
- **Billing**: `get_credit_model(user_id, org_id)` routes to
  `OrgCreditModel` when an org is active; executor charging follows
- **Resources**: graphs, executions, API keys, chat sessions, store
  listings carry `organizationId` / `teamId`; marketplace listings support
  org ownership and cross-org transfers
- **Frontend**: org/team switcher in the navbar, `OrgTeamProvider`
  bootstraps context on login, zustand store persists the active org/team
- **Migration**: idempotent personal-org bootstrap
  (`create_orgs_for_existing_users`) behind a Redis lock at startup

Deferred from the foundation series: seat enforcement (paywall/subscription
gating work owns this) and the NOT-NULL cutover migration for
`organizationId` columns.

## 1. Governance & policy controls

The headline ask: org/team admins control what members can do and which
capabilities agents may use.

| Feature | Sketch |
| --- | --- |
| **Block enable/disable per org/team** | New `BlockPolicy` table (`orgId`, `teamId?`, `blockId`, `allowed`, optional reason). Enforcement at three choke points: `get_graph_blocks` (filter the palette), `validate_graph` (reject save/activation of graphs using denied blocks), and executor node resolution (defense-in-depth refusal at run time, since policies can change after a graph is saved). Admin UI under org settings; team policy overrides org policy only toward more-restrictive. |
| **Provider/credential policy** | Same pattern keyed on provider name instead of block ID: restrict which integrations (HTTP request, code execution, specific SaaS providers) teams may attach credentials for. `IntegrationCredential` already carries an org relation to hang this on. |
| **Model allowlist + per-team spending caps** | Policy keyed on LLM model family; enforced where block costs resolve (`block_usage_cost`) and in copilot model selection. Caps check `OrgCreditTransaction` aggregates per team before dispatch. |
| **Sensitive-action approval routing** | `ExecutionContext.sensitive_action_safe_mode` and human-in-the-loop review already exist per-user; org version routes pending reviews to org/team admins instead of only the owner, using the existing review queue. |
| **Safe-mode defaults per org** | Org `settings` JSON already exists on `Organization`; add admin-managed defaults that seed `human_in_the_loop_safe_mode` / `sensitive_action_safe_mode` for all member executions, member-overridable only if policy allows. |

## 2. Identity & access

| Feature | Sketch |
| --- | --- |
| **SSO (SAML/OIDC)** | Supabase auth supports external IdPs; org-level config maps IdP groups → teams. `OrganizationAlias` (domain aliases) is the natural anchor for domain-capture ("anyone @acme.com joins Acme org"). |
| **SCIM provisioning** | Sync engine writing to `OrgMember`/`TeamMember`; deprovisioning must also revoke API keys and seats (`OrganizationSeatAssignment`). |
| **Custom roles / RBAC** | Today roles are three booleans (`isOwner`, `isAdmin`, `isBillingManager`) on both org and team membership. A `Role` table with permission grants replaces the booleans when customers need finer slicing (e.g. "can publish to marketplace but not manage billing"). `RequestContext` already centralizes the flag derivation, so the cutover is contained. |
| **Audit log surfacing** | `AuditLog` model exists with org relation; needs consistent write coverage across mutating routes (member changes, policy changes, transfers, key creation/revocation) and an admin-facing viewer with filtering/export. |
| **Session & network policy** | Org-enforced re-auth interval, IP allowlists for API-key traffic (check at `validate_api_key` where org context is already resolved). |

## 3. Resource management & collaboration

| Feature | Sketch |
| --- | --- |
| **Team-shared agent library** | `AgentGraph.teamId` exists; the library views currently scope by user. Team-visible library = library queries keyed by team membership plus `visibility` (enum already on graphs/library agents/chat sessions). |
| **Org-shared credentials** | `IntegrationCredential` already relates to `Organization`; needs scoping rules (org-owned credentials usable by member teams), admin management UI, and executor credential resolution preferring team → org → user. |
| **Environment promotion (dev → prod teams)** | `TransferRequest` covers cross-org moves; intra-org promotion is a lighter copy with provenance (`forkedFromId` already tracked on graphs). |
| **Quotas per team** | Run-concurrency and storage ceilings checked in `add_graph_execution` (org/team already threaded there) and workspace upload paths. |
| **Shared chat / handoff** | Chat sessions carry `organizationId`/`teamId` and a sharing layer (share tokens, linked execution shares) exists; team-visible sessions are a `visibility` policy on top. |

## 4. Billing & finance

| Feature | Sketch |
| --- | --- |
| **Seat management & enforcement** | Schema shipped (`OrganizationSubscription`, `OrganizationSeatAssignment`, `seat_status` on `RequestContext`); enforcement intentionally deferred to the paywall/subscription workstream. |
| **Budgets & alerts per team** | Aggregate `OrgCreditTransaction` by team; thresholds trigger the existing notification pipeline (low-balance handling already has a user-level analog in the executor). |
| **Cost attribution dashboards** | Transactions already record user, team, graph, and execution metadata — the work is aggregation endpoints + admin UI, not new write paths. |
| **Org auto-top-up** | `Organization.topUpConfig` field exists; needs the org analog of `set_auto_top_up` plus Stripe customer on org (`stripeCustomerId` present). |
| **Invoicing (PO / net terms / tax IDs)** | Stripe invoice plumbing per org customer; `OrgCreditModel.list_invoices` is stubbed for this. |

## 5. Marketplace

| Feature | Sketch |
| --- | --- |
| **Org-private marketplace** | Store listings already support `owningOrgId`; an org-internal visibility tier reuses the listing/version review flow with org-scoped queries instead of the public `StoreAgent` view. |
| **Publishing approval chain** | Submission review exists for platform admins; org version inserts an org-admin approval step before a listing leaves the org boundary. |
| **Listing transfers** | Shipped (`TransferRequest` source/target org with admin approval on both ends). |

## 6. Observability & compliance

| Feature | Sketch |
| --- | --- |
| **Org-wide run history & analytics** | Executions carry org/team; admin endpoints mirror the existing per-user execution listings with org-role authorization. |
| **Data retention policy** | Org `settings`-driven TTL sweeping executions/chat sessions; deletion paths (cascade rules on shares, workspaces) already exist to build on. |
| **Export / portability** | Org-scoped bulk export of graphs (graph export logic exists per-graph), execution history, and transactions. |
| **Compliance evidence** | Once `AuditLog` coverage is complete, scheduled export bundles cover the SOC2-style asks. |

## Suggested build order

1. **Block/provider policy** (per-team enable/disable) — highest admin demand,
   clear enforcement choke points, no schema risk beyond one table
2. **Audit log write coverage + viewer** — prerequisite trust layer for
   everything else admins touch
3. **Team-shared library + org credentials** — turns tenancy from isolation
   into collaboration; biggest day-to-day user value
4. **Budgets/caps + cost attribution** — natural follow-on to org billing
   already in place
5. **RBAC generalization** — only when boolean roles demonstrably block a
   customer; cutover is contained in `get_request_context`
6. **SSO/SCIM** — enterprise gate; heavy lift, schedule against real demand
