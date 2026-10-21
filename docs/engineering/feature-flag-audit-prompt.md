# Tuesday feature-flag retirement and default audit

Use the prompt below for a weekly Tuesday task in AutoGPT. Set the run time in the task schedule using America/Chicago. This is an audit prompt; it does not create a schedule or authorize production changes.

```text
Audit AutoGPT's feature flags and code defaults. Find product features that have been fully rolled out for at least seven consecutive days and still carry obsolete rollout branches or fall back to older behavior when the flag service is absent or unavailable.

Sources and scope:
- Read the production PostHog project identified in Significant-Gravitas/AutoGPT's docs/engineering/feature-flags.md; confirm the project before querying. Use connected, authorized tools and paginate all flag and activity results. Report missing access instead of guessing.
- Inspect a fresh upstream dev revision. Record its exact SHA and the audit timestamp. Use the previous Tuesday report and snapshot, if available, to describe changes; they are context, not current truth. Explicitly report when previous-run artifacts are unavailable.

For each candidate:
1. Read its full definition: active state, ordered targeting rules, exceptions, cohorts, variants/payloads, holdouts and dependencies. A 100% employee cohort is not 100% of everyone. An API-key owner's evaluated flags do not represent all employees or all users. Confirm the effective result for the whole intended production audience, including authenticated/anonymous differences where relevant.
2. Use audit history to establish how long that effective state has been continuous. Track both targeting changes and payload changes. Include changes to referenced cohorts, dependent flags and holdouts where they affect the result. Matching weekly snapshots alone do not prove uninterrupted rollout; record gaps in history as unproven. Creation time alone is not an enablement date. Imported flags may only establish an earliest confirmed boundary; say so. If history cannot establish seven days, classify age as unproven. Highlight 14-day and 30-day stability separately. A rollback or meaningful configuration change restarts the relevant clock.
3. Trace the key through frontend defaults, backend helpers, decorators/dependencies, API routes, tools, workers, schedulers and callsite-specific fallback arguments. Compare actual behavior when the provider is missing, fails, returns no definition, or explicitly returns false. Check SDK contracts: a successful response omitting an inactive flag may look identical to a failed read through a convenience helper. Distinguish a code fallback from an authoritative answer. Identify build-time overrides and deployment settings separately. Cite exact files and lines at the recorded SHA.
4. Compare structured payloads semantically. An empty model/cost map can mean “use existing defaults.” A variant name is not a boolean. List real array/object mismatches separately and redact secrets or account-specific identifiers.
5. Classify: suitable flag-retirement candidate; default-update candidate with a justified continuing rollout control; recent or restricted rollout; intentional deployment configuration; operational/enforcement control needing review; already aligned; or insufficient evidence. Keep billing, Stripe/trials, entitlements, authentication/security, provider requirements and background model costs separate. A longstanding rollout is not proof that removing its control is safe. Retirement means removing the flag consumer and obsolete branches, not changing the default to true. Preserve authorization and user preference checks.
6. Check existing open and recently merged PRs/issues before recommending work. Link matching work and avoid duplicate recommendations. For each remaining candidate, propose the smallest coordinated change and meaningful tests: absent/failed provider, successful inactive-flag omission, explicit false, unchanged nonselected flags, frontend/backend parity, authority metadata and relevant worker/route behavior.

Output:
- A concise verdict and what changed since the previous audit.
- A table with key/type, configured production value and audience, confirmed continuous rollout age, frontend/backend defaults, deployment overrides, mismatch, recommendation and evidence links.
- Separate recent/employee-only flags and billing/operational exceptions. Clearly state unknowns; say “no new candidates” when appropriate.
- Save feature-flag-audit-YYYY-MM-DD.md and feature-flag-audit-YYYY-MM-DD.json as task artifacts for the next Tuesday run, with a retrievable reference to the latest report and snapshot. Include the revision, project, timestamp, flag IDs/versions, sanitized effective values/payloads, normalized targeting fingerprints, code fallback values, history coverage, evidence boundary and recommendation status, without credentials or personal data.

Do not change PostHog, production settings, subscriptions, deployments or code; do not open/merge PRs, create tickets, or notify other people unless separately authorized. Produce PR-ready recommendations for review. If a required tool is unavailable, finish the portions supported by evidence and identify the exact blocker.
```
