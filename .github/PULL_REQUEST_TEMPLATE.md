### Flag-off behavior 🚩

**Does this change intentionally alter flag-off behavior?** If yes, explain why
and describe the expected impact on the production flag-off contract.

- The system-prompt SHA-256 hash (locked in `_PRE_CHANGE_PROMPT_SHA256`) must
  not change unless the `_CACHEABLE_SYSTEM_PROMPT` constant is intentionally
  updated.
- The `/team` page must continue to `notFound()` when `HIRE_EXPERTS` is off.
- The sessions list must stay flat (no flag-gated grouping or nesting) when
  all flags are off.

---

### Why / What / How

<!-- Why: Why does this PR exist? What problem does it solve, or what's broken/missing without it? -->
<!-- What: What does this PR change? Summarize the changes at a high level. -->
<!-- How: How does it work? Describe the approach, key implementation details, or architecture decisions. -->

### Changes 🏗️

<!-- List the key changes. Keep it higher level than the diff but specific enough to highlight what's new/modified. -->

### Checklist 📋

#### For code changes:
- [ ] I have clearly listed my changes in the PR description
- [ ] I have made a test plan
- [ ] I have tested my changes according to the test plan:
  <!-- Put your test plan here: -->
  - [ ] ...

<details>
  <summary>Example test plan</summary>
  
  - [ ] Create from scratch and execute an agent with at least 3 blocks
  - [ ] Import an agent from file upload, and confirm it executes correctly
  - [ ] Upload agent to marketplace
  - [ ] Import an agent from marketplace and confirm it executes correctly
  - [ ] Edit an agent from monitor, and confirm it executes correctly
</details>

#### For configuration changes:

- [ ] `.env.default` is updated or already compatible with my changes
- [ ] `docker-compose.yml` is updated or already compatible with my changes
- [ ] I have included a list of my configuration changes in the PR description (under **Changes**)

<details>
  <summary>Examples of configuration changes</summary>

  - Changing ports
  - Adding new services that need to communicate with each other
  - Secrets or environment variable changes
  - New or infrastructure changes such as databases
</details>
