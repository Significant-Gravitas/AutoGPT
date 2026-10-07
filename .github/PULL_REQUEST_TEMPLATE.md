### Why / What / How

<!-- Why: Why does this PR exist? What problem does it solve, or what's broken/missing without it? -->
<!-- What: What does this PR change? Summarize the changes at a high level. -->
<!-- How: How does it work? Describe the approach, key implementation details, or architecture decisions. -->

### Changes 🏗️

<!-- List the key changes. Keep it higher level than the diff but specific enough to highlight what's new/modified. -->

### Agents and large language models used

<!-- List each agent platform with its model name/version (e.g. Claude Code with Claude Opus 4.1).
Write None if no agents were used, or unknown for unavailable model details. -->

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

#### For frontend UI changes (`autogpt_platform/frontend/src`):
<!-- See autogpt_platform/frontend/DESIGN.md. Delete this section if the PR has no UI changes. -->
- [ ] Uses design tokens and the atoms, molecules and organisms in `src/components` (`Text`, `Button`, `Link`, `Icon`...), not raw elements or one-off styles
- [ ] No new imports from `src/components/__legacy__` or `src/components/ui` outside `src/components`, and no new entries in `eslint-allowlist.json`
- [ ] No `dark:` classes, no hex colour classes, no default-palette families (`gray`, `neutral`, `amber`, `violet`...)
- [ ] Added or updated a story for every design-system component this PR adds or changes

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
