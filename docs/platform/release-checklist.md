# Release Checklist

## Pre-release steps

### Flag-off smoke run

Run the flag-off regression tests before every release to verify all LaunchDarkly
flags evaluate to their safe defaults and core flows stay intact.

```bash
# Backend — system-prompt hash lock + prompt cache fallback
cd autogpt_platform/backend
poetry run pytest backend/copilot/expert_context_test.py \
  backend/copilot/prompt_cache_test.py -v

# Frontend — flag defaults, /team 404, and unit suite
cd ../frontend
pnpm test:unit

# Frontend — E2E smoke (copilot, marketplace, library)
pnpm exec playwright test --grep="flag-off"
```

All three suites must pass green on the `main` branch before a release is cut.

### CI flag-off gate

The `Flag-off regression` workflow (`.github/workflows/flag-off-regression.yml`)
runs on every push and PR to `main` and initiative branches
(`**/feature/**`, `**/initiative/**`, `**/epic/**`). Confirm it is green on the
release commit.

### LaunchDarkly flag audit

Review all flags in the [LaunchDarkly dashboard](https://app.launchdarkly.com/)
to confirm no flag was left in a perma-on state that would diverge flag-off
behaviour from what the CI runs test.

### Release notes

If any flag-off invariant intentionally changed (system prompt hash updated,
/team 404 removed, session list restructured), call it out explicitly in the
release notes with a rationale.
