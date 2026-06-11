export const meta = {
  name: 'contribute-block',
  description: 'Implement AutoGPT block(s) end-to-end and open a PR per request: select → plan → implement → review → verify → PR',
  whenToUse:
    'Maintainer-facing batch automation of the block pipeline. Pass a block request (plain language is fine), a list of requests, or a bare {count: N} to auto-pick from open `platform/blocks` issues. One branch + one PR per request, run sequentially. Interactive single-block work should use the /block-implementer skill instead (unbounded review loops); this workflow bounds its loops and ships unclean items as draft PRs so one stuck item cannot stall the batch. The workflow ends with a /pr-polish handoff list — run /pr-polish per PR in the foreground afterward (it cannot run inside workflow agents).',
  phases: [
    { title: 'Select', detail: 'resolve explicit request(s) or auto-pick from open platform/blocks issues' },
    { title: 'Plan', detail: 'feature-planner + add-block checklist, review-feature-plan loop (max 3 rounds)' },
    { title: 'Implement', detail: 'branch off origin/dev + implementation-executor agent, test-first' },
    { title: 'Review', detail: 'review-impl loop (max 3) + fresh-context block cross-check (max 2)' },
    { title: 'Verify', detail: 'format, lint, per-block test, full block registry test — serialized' },
    { title: 'PR', detail: 'pathspec commit, push, PR against dev; draft when any gate is unclean' },
    { title: 'Status', detail: 'read-only CI sweep across opened PRs; emit /pr-polish handoff list' },
  ],
}

// Bounded loops by design: this is batch automation. A request that exhausts its
// rounds degrades to a DRAFT PR with findings documented, instead of blocking
// the batch. The interactive /block-implementer skill keeps the unbounded
// loop-until-clean semantics.
const MAX_PLAN_REVIEW_ROUNDS = 3
const MAX_IMPL_REVIEW_ROUNDS = 3
const MAX_CROSSCHECK_ROUNDS = 2
const MAX_VERIFY_RETRIES = 2

const REPO = 'Significant-Gravitas/AutoGPT'
const BACKEND_DIR = 'autogpt_platform/backend'

// args: bare request string | { request? , requests?: string[], count? }.
function normalizeArgs(a) {
  if (typeof a === 'string') {
    const r = a.trim()
    return { requests: r ? [r] : [], count: 1 }
  }
  if (a && typeof a === 'object') {
    if (Array.isArray(a.requests) && a.requests.length) {
      return { requests: a.requests.map((r) => String(r).trim()).filter(Boolean), count: 0 }
    }
    if (typeof a.request === 'string' && a.request.trim()) {
      return { requests: [a.request.trim()], count: 0 }
    }
    const count = Number.isInteger(a.count) && a.count > 0 ? a.count : 1
    return { requests: [], count }
  }
  return { requests: [], count: 1 }
}

const WORKLIST_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['items'],
  properties: {
    items: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['request'],
        properties: {
          request: { type: 'string', description: 'Plain-language block request' },
          issue: { type: 'number', description: 'GitHub issue number, when sourced from the issue queue' },
        },
      },
    },
  },
}

const BRANCH_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['branch'],
  properties: {
    branch: { type: 'string', description: 'Exact branch name created (feature/block-<slug>[-N])' },
  },
}

const REVIEW_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['clean', 'findings'],
  properties: {
    clean: { type: 'boolean', description: 'true when there are no blocking findings' },
    findings: { type: 'array', items: { type: 'string' } },
  },
}

const IMPL_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['filesChanged', 'blockNames', 'envVars', 'scopeExpansion'],
  properties: {
    filesChanged: { type: 'array', items: { type: 'string' }, description: 'Paths only, including tests and snapshots' },
    blockNames: { type: 'array', items: { type: 'string' }, description: 'Block class names created or modified' },
    envVars: { type: 'array', items: { type: 'string' }, description: 'New env vars a deployer must set (empty if none)' },
    scopeExpansion: { type: 'string', description: 'One-line description of scope growth, or the literal "None."' },
  },
}

const CROSSCHECK_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['clean', 'findings'],
  properties: {
    clean: { type: 'boolean' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['category', 'detail'],
        properties: {
          category: {
            type: 'string',
            enum: ['mock-fidelity', 'credentials-e2e', 'schema-rules', 'ssrf-requests', 'composability', 'ui-copy', 'test-adequacy'],
          },
          location: { type: 'string', description: 'file:line if known' },
          detail: { type: 'string' },
        },
      },
    },
  },
}

const VERIFY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['passed', 'commands', 'failures'],
  properties: {
    passed: { type: 'boolean' },
    commands: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['name', 'status'],
        properties: { name: { type: 'string' }, status: { type: 'string' } },
      },
    },
    failures: { type: 'array', items: { type: 'string' } },
  },
}

const PR_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['opened'],
  properties: {
    opened: { type: 'boolean' },
    prUrl: { type: 'string' },
    draft: { type: 'boolean' },
  },
}

const STATUS_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['statuses'],
  properties: {
    statuses: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['prUrl', 'ci'],
        properties: {
          prUrl: { type: 'string' },
          ci: { type: 'string', enum: ['pass', 'fail', 'pending', 'mixed', 'unknown'] },
          notes: { type: 'string', description: 'Failing/pending check names, early bot findings' },
        },
      },
    },
  },
}

function planPrompt(request) {
  return (
    `Use the \`feature-planner\` skill (read \`.claude/skills/feature-planner/SKILL.md\`) ` +
    `to produce an implementation plan for this AutoGPT block request:\n\n"${request}"\n\n` +
    `MANDATORY: first read \`.claude/skills/add-block/SKILL.md\` and structure the plan ` +
    `around its checklist. The plan must name: provider folder (new or existing) and ` +
    `\`_config.py\` auth/cost setup; exact block class name(s), one action each; the ` +
    `test triple including WHERE the real API response shape comes from (docs URL or ` +
    `live call) so mocks are honest; OAuth/webhook wiring when applicable; and a ` +
    `composability statement (which existing blocks feed its inputs / consume its ` +
    `outputs). Return the full plan text.`
  )
}

function reviewPlanPrompt(request, plan) {
  return (
    `Use the \`review-feature-plan\` skill (read \`.claude/skills/review-feature-plan/SKILL.md\`) ` +
    `to review this block implementation plan. Additionally check it against the ` +
    `registration-rules table in \`.claude/skills/add-block/SKILL.md\`; a mock shape ` +
    `with no cited source is a finding. Set clean=true only with zero blocking ` +
    `findings.\n\nREQUEST: ${request}\n\nPLAN:\n${plan}`
  )
}

function replanPrompt(request, plan, findings) {
  return (
    `Revise this block implementation plan to address every review finding. Re-read ` +
    `\`.claude/skills/add-block/SKILL.md\` first. Return the full revised plan text.\n\n` +
    `REQUEST: ${request}\n\nFINDINGS:\n` +
    findings.map((f) => `- ${f}`).join('\n') +
    `\n\nCURRENT PLAN:\n${plan}`
  )
}

function implementPrompt(request, plan) {
  return (
    `Implement this AutoGPT block request on the current branch, following the ` +
    `approved plan exactly. Read \`.claude/skills/add-block/SKILL.md\` before editing.\n\n` +
    `REQUEST: ${request}\n\n` +
    `Test-first: write the failing test, confirm it fails, implement, confirm it ` +
    `passes. Run verification commands from \`${BACKEND_DIR}\` and SERIALLY — never ` +
    `two pytest runs at once (they exhaust DB connection slots). Do not ask for ` +
    `clarification — on ambiguity take the architecturally idiomatic path and note ` +
    `it. If scope expands (new provider infra, OAuth handler, webhook manager), ` +
    `proceed and set scopeExpansion to a one-line description, else the literal ` +
    `"None.". Document any new env vars in \`${BACKEND_DIR}/.env.default\`.\n\n` +
    `Do NOT commit — leave changes in the working tree for review. List ` +
    `filesChanged (paths only), blockNames, and envVars.\n\nAPPROVED PLAN:\n${plan}`
  )
}

function reviewImplPrompt(request) {
  return (
    `Use the \`review-impl\` skill (read \`.claude/skills/review-impl/SKILL.md\`) against ` +
    `the current uncommitted working-tree diff for this block request: "${request}". ` +
    `The Blocks surface lens is mandatory. Set clean=true only if there are no ` +
    `defects, gaps, or missing cases. List each finding as a concrete string with ` +
    `file:line.`
  )
}

function fixImplPrompt(request, findings) {
  return (
    `Address every one of these review findings for the block request "${request}" ` +
    `with code changes in the working tree. Re-run the targeted block test after ` +
    `fixing (serially). Do not commit.\n\nFINDINGS:\n` +
    findings.map((f) => `- ${f}`).join('\n')
  )
}

function crossCheckPrompt(request) {
  return (
    `You are an INDEPENDENT reviewer with fresh context. You are given ONLY the ` +
    `unified diff (\`git diff\`), the repo AGENTS.md files, and the skills under ` +
    `\`.claude/skills/\`. Ignore any prior conversation. Review the uncommitted ` +
    `change for the block request "${request}" and check ALL of:\n` +
    `(a) mock-fidelity — does every test_mock return the REAL provider API response ` +
    `shape (verify against the provider's API docs), not a convenient invention?\n` +
    `(b) credentials-e2e — _config.py provider → credentials field → test ` +
    `credentials → (OAuth) handler registered in integrations/oauth/__init__.py + ` +
    `Secrets env vars in util/settings.py?\n` +
    `(c) schema-rules — UUID4 id (36 chars, not copied), class name ends in Block, ` +
    `every field a SchemaField, boolean inputs have defaults, error output is str?\n` +
    `(d) ssrf-requests — all HTTP via backend.util.request.Requests; flag raw ` +
    `httpx/requests/aiohttp for user-influenced URLs?\n` +
    `(e) composability — inputs/outputs connect to plausible neighbor blocks in the ` +
    `graph editor; flag dead-end output types?\n` +
    `(f) ui-copy — block description and every SchemaField description read like ` +
    `product copy for a non-technical builder user?\n` +
    `(g) test-adequacy — a test exists that fails without the change; failure paths ` +
    `exercised, not just happy path?\n` +
    `Set clean=true only if NONE of (a)-(g) produced a finding. Categorize each finding.`
  )
}

function fixCrossCheckPrompt(request, findings) {
  return (
    `A fresh-context reviewer found these issues in the "${request}" block change. ` +
    `Fix each with code in the working tree. Do not commit.\n\nFINDINGS:\n` +
    findings
      .map((f) => `- [${f.category}] ${f.location || ''} ${f.detail}`)
      .join('\n')
  )
}

function verifyPrompt(request, blockNames) {
  const perBlock = (blockNames || [])
    .map((b) => `poetry run pytest 'backend/blocks/test/test_block.py::test_available_blocks[${b}]' -xvs`)
    .join(' ; then ')
  return (
    `Run block verification for "${request}" from \`${BACKEND_DIR}\`, in this exact ` +
    `order, STRICTLY ONE COMMAND AT A TIME (concurrent pytest exhausts DB ` +
    `connection slots), fixing in-loop on failure (max ${MAX_VERIFY_RETRIES} retries ` +
    `per command) before continuing:\n` +
    `1. poetry run format\n` +
    `2. poetry run lint\n` +
    `3. ${perBlock || 'poetry run pytest backend/blocks/test/test_block.py -x'}\n` +
    `4. poetry run pytest backend/blocks/test/test_block.py -x   (full registry: ` +
    `IDs, schemas, all blocks still load)\n` +
    `Set passed=true only if every command is clean. Record each command's status; ` +
    `list any unresolved failures in failures[]. If a failure is clearly unrelated ` +
    `to this diff, record it as "(unrelated) ..." and do not chase it.`
  )
}

function prPrompt(request, { impl, verify, partialReasons }) {
  const partial = partialReasons.length > 0
  const verifyLines = (verify.commands || [])
    .map((c) => `  - \`${c.name}\` — ${c.status}`)
    .join('\n')
  return (
    `Commit and open a PR for the block request "${request}".\n\n` +
    `1. Stage by pathspec ONLY (never \`git add -A\`): \`git status --short\`, confirm ` +
    `every file belongs to this change, then \`git add <paths>\` using this list plus ` +
    `any test/snapshot files it generated:\n` +
    (impl.filesChanged || []).map((f) => `   - ${f}`).join('\n') +
    `\n2. Commit: \`git commit -m "feat(blocks): ${(impl.blockNames || []).join(', ') || request}"\` ` +
    `(adjust to a clean conventional-commit summary under 72 chars).\n` +
    `3. Push: verify HEAD is on the expected branch, then \`git push -u origin HEAD\`.\n` +
    `4. Open the PR against dev with --body-file (NEVER inline --body):\n` +
    `   PR_BODY=$(mktemp); write the body to it; ` +
    `\`gh pr create --repo ${REPO} --base dev ${partial ? '--draft ' : ''}--title "<same conventional title>" --body-file "$PR_BODY"\`; rm it.\n` +
    `   Fill .github/PULL_REQUEST_TEMPLATE.md with a Why / What / How structure.\n` +
    `   Include sections: "Blocks added" (${(impl.blockNames || []).join(', ') || 'n/a'}), ` +
    `"Env vars" (${(impl.envVars || []).length ? impl.envVars.join(', ') : 'none'}), ` +
    `"Scope expansion" (${impl.scopeExpansion || 'None.'}), and "Verification":\n${verifyLines}\n` +
    (partial
      ? `   This run is PARTIAL — the PR MUST be a draft and the body MUST lead with an ` +
        `"Unresolved findings" section listing:\n` +
        partialReasons.map((r) => `   - ${r}`).join('\n') + `\n`
      : '') +
    `Return opened=true, the prUrl, and draft=${partial}.`
  )
}

async function selectItems({ requests, count }) {
  if (requests.length) return requests.map((request) => ({ request }))
  const res = await agent(
    `Pick the ${count} best block request(s) to implement from the GitHub issue ` +
      `queue. Run: gh issue list --repo ${REPO} --state open --label "platform/blocks" ` +
      `--limit 50 --json number,title,body,reactionGroups. Selection criteria: a ` +
      `clear "add <service> block(s)" request (not a bug in an existing block), no ` +
      `open PR already linked, highest community reactions then oldest first. For ` +
      `each pick, return request as a one-sentence plain-language description built ` +
      `from the issue title+body, and the issue number.`,
    // Issue-queue judgment, but low-stakes (a bad pick just wastes one item) — mid tier.
    { label: 'select-blocks', phase: 'Select', schema: WORKLIST_SCHEMA, model: 'sonnet' },
  )
  return (res && res.items ? res.items : []).slice(0, count)
}

async function createBranch(request) {
  const res = await agent(
    `Create a git branch for the block request "${request}". Require a clean ` +
      `working tree (\`git status --short\` empty — if not, STOP and report). Build ` +
      `slug="feature/block-<lowercase-hyphenated-short-name>". Collision guard: if ` +
      `the branch exists locally (git rev-parse --verify) or on origin ` +
      `(git ls-remote --exit-code origin), append -2, -3, ... until free. Then: ` +
      `git fetch origin dev --quiet && git checkout -b "$slug" origin/dev. Return ` +
      `the exact branch name created.`,
    // Mechanical git commands — cheapest tier.
    { label: `branch:${request.slice(0, 40)}`, phase: 'Implement', schema: BRANCH_SCHEMA, model: 'haiku' },
  )
  return res && res.branch ? res.branch : null
}

async function planItem(request) {
  let plan = await agent(planPrompt(request), {
    label: `plan:${request.slice(0, 40)}`,
    phase: 'Plan',
  })
  let planClean = false
  for (let round = 1; round <= MAX_PLAN_REVIEW_ROUNDS; round++) {
    const review = await agent(reviewPlanPrompt(request, plan), {
      label: `review-plan#${round}`,
      phase: 'Plan',
      schema: REVIEW_SCHEMA,
    })
    if (review.clean) {
      planClean = true
      break
    }
    plan = await agent(replanPrompt(request, plan, review.findings), {
      label: `replan#${round}`,
      phase: 'Plan',
    })
  }
  return { plan, planClean }
}

async function implementItem(request, plan) {
  return await agent(implementPrompt(request, plan), {
    label: `implement:${request.slice(0, 40)}`,
    phase: 'Implement',
    agentType: 'implementation-executor',
    schema: IMPL_SCHEMA,
  })
}

async function reviewImpl(request) {
  for (let round = 1; round <= MAX_IMPL_REVIEW_ROUNDS; round++) {
    const review = await agent(reviewImplPrompt(request), {
      label: `review-impl#${round}`,
      phase: 'Review',
      schema: REVIEW_SCHEMA,
    })
    if (review.clean) return true
    await agent(fixImplPrompt(request, review.findings), {
      label: `fix-impl#${round}`,
      phase: 'Review',
      agentType: 'implementation-executor',
    })
  }
  return false
}

async function crossCheck(request) {
  let res = await agent(crossCheckPrompt(request), {
    label: 'crosscheck',
    phase: 'Review',
    schema: CROSSCHECK_SCHEMA,
  })
  for (
    let round = 1;
    round <= MAX_CROSSCHECK_ROUNDS && !res.clean && res.findings && res.findings.length;
    round++
  ) {
    await agent(fixCrossCheckPrompt(request, res.findings), {
      label: `fix-crosscheck#${round}`,
      phase: 'Review',
      agentType: 'implementation-executor',
    })
    res = await agent(crossCheckPrompt(request), {
      label: `recheck#${round}`,
      phase: 'Review',
      schema: CROSSCHECK_SCHEMA,
    })
  }
  return res
}

phase('Select')
const { requests, count } = normalizeArgs(args)
const items = await selectItems({ requests, count })
log(`Work-list (${items.length}): ${items.map((i) => i.request).join(' | ') || '(none)'}`)

// Items run SEQUENTIALLY on purpose: they share one working tree, and the
// backend test suite cannot run concurrently (DB connection slots).
const summary = []
for (const item of items) {
  const { request } = item
  try {
    const branch = await createBranch(request)
    if (!branch) throw new Error('branch creation failed')

    const { plan, planClean } = await planItem(request)
    const impl = await implementItem(request, plan)
    const implReviewClean = await reviewImpl(request)
    const cross = await crossCheck(request)
    const verify = await agent(verifyPrompt(request, impl && impl.blockNames), {
      label: `verify:${request.slice(0, 40)}`,
      phase: 'Verify',
      schema: VERIFY_SCHEMA,
    })

    const partialReasons = []
    if (!planClean) partialReasons.push(`plan review not clean after ${MAX_PLAN_REVIEW_ROUNDS} rounds`)
    if (!implReviewClean) partialReasons.push(`impl review not clean after ${MAX_IMPL_REVIEW_ROUNDS} rounds`)
    if (!cross.clean)
      partialReasons.push(
        ...(cross.findings || []).map((f) => `[${f.category}] ${f.location || ''} ${f.detail}`),
      )
    if (!verify.passed) partialReasons.push(...(verify.failures || []))

    const pr = await agent(prPrompt(request, { impl, verify, partialReasons }), {
      label: `pr:${request.slice(0, 40)}`,
      phase: 'PR',
      schema: PR_SCHEMA,
    })

    const status = partialReasons.length ? 'partial (draft PR)' : 'success'
    summary.push({
      request,
      issue: item.issue || null,
      branch,
      blocks: (impl && impl.blockNames) || [],
      envVars: (impl && impl.envVars) || [],
      prUrl: pr && pr.prUrl ? pr.prUrl : null,
      status,
    })
    log(`${request}: ${status}${pr && pr.prUrl ? ' -> ' + pr.prUrl : ''}`)
  } catch (e) {
    summary.push({
      request,
      issue: item.issue || null,
      branch: null,
      blocks: [],
      envVars: [],
      prUrl: null,
      status: 'aborted',
    })
    log(`${request}: aborted -- ${e && e.message ? e.message : 'error'}`)
  }
}

// Read-only CI sweep. By the time the last item's PR is open, CI on the
// earlier PRs has had time to run — one cheap status pass makes the summary
// actionable without blocking on CI inside the per-item loop.
phase('Status')
const opened = summary.filter((s) => s.prUrl)
let statuses = []
if (opened.length) {
  const res = await agent(
    `For each of these PRs, run \`gh pr checks <url> --repo ${REPO} --json name,state,bucket\` ` +
      `and \`gh pr view <url> --repo ${REPO} --json reviews --jq '[.reviews[] | select(.body != "")] | length'\`. ` +
      `READ-ONLY — do not push, comment, or modify anything. Report ci as: pass ` +
      `(all buckets pass/skipping), fail (any fail/cancel), pending (any pending, ` +
      `none failing), mixed (failing + pending), unknown (query failed). Put ` +
      `failing/pending check names and any early bot-review count in notes.\n\nPRs:\n` +
      opened.map((s) => `- ${s.prUrl}`).join('\n'),
    // Read-only gh status queries — cheapest tier.
    { label: 'ci-sweep', phase: 'Status', schema: STATUS_SCHEMA, model: 'haiku' },
  )
  statuses = (res && res.statuses) || []
}

// /pr-polish (which loops /pr-review + /pr-address to merge-ready) cannot run
// inside workflow agents — they don't inherit the Skill registry, and pr-polish
// itself documents it must run in the foreground main thread. Hand it off.
const nextSteps = opened.map((s) => `/pr-polish ${s.prUrl}`)
if (nextSteps.length) {
  log(`Polish handoff — run in the foreground, one at a time: ${nextSteps.join(' ; ')}`)
}

return { summary, ciStatuses: statuses, nextSteps }
