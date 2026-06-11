export const meta = {
  name: 'sweep-prs',
  description: 'Read-only mass PR triage: classify the open queue in parallel, cluster duplicates, adversarially verify every actionable verdict, return an action plan',
  whenToUse:
    'The analysis arm of /pr-sweep, for queues too big to classify inline (~20+ PRs). Strictly read-only — it returns an action plan (auto-close / recommend-close / duplicate clusters with winners / deep-review candidates); the /pr-sweep skill executes mutations afterward with its authority gates. Pass {scope?: "<gh search terms>", limit?: N}.',
  phases: [
    { title: 'Snapshot', detail: 'list open PRs against dev with stats' },
    { title: 'Classify', detail: 'batched parallel classification, ~10 PRs per agent' },
    { title: 'Cluster', detail: 'barrier: group duplicate/overlap candidates across the whole queue' },
    { title: 'Verify', detail: 'adversarial refuters on every auto-close verdict and cluster; default keep' },
  ],
}

const REPO = 'Significant-Gravitas/AutoGPT'
const BATCH_SIZE = 10

const SNAPSHOT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['prs'],
  properties: {
    prs: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['number', 'title', 'author'],
        properties: {
          number: { type: 'number' },
          title: { type: 'string' },
          author: { type: 'string' },
          isDraft: { type: 'boolean' },
          additions: { type: 'number' },
          deletions: { type: 'number' },
          changedFiles: { type: 'number' },
          daysSinceUpdate: { type: 'number' },
          assigned: { type: 'boolean' },
        },
      },
    },
  },
}

const CLASSIFY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['classifications'],
  properties: {
    classifications: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['number', 'category', 'confidence', 'evidence'],
        properties: {
          number: { type: 'number' },
          category: {
            type: 'string',
            enum: ['auto-close', 'recommend-close', 'duplicate-candidate', 'needs-deep-review', 'keep'],
          },
          closeReason: {
            type: 'string',
            enum: ['spam', 'blank-template', 'superseded', 'docs-churn', 'refactor-drive-by', 'test-only-no-bug', 'stale-abandoned', 'dirty-branch'],
            description: 'Required when category is auto-close or recommend-close. Only spam/blank-template/superseded are valid for auto-close.',
          },
          confidence: { type: 'string', enum: ['clear', 'likely', 'uncertain'] },
          evidence: { type: 'string', description: 'Specific, checkable evidence — for superseded, the merged PR/commit that ships the same change' },
          topics: { type: 'array', items: { type: 'string' }, description: '2-4 keywords describing what the PR does, for cross-queue clustering' },
          keyFiles: { type: 'array', items: { type: 'string' }, description: 'Up to 5 most significant changed paths' },
        },
      },
    },
  },
}

const CLUSTER_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['clusters'],
  properties: {
    clusters: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['topic', 'members', 'rationale'],
        properties: {
          topic: { type: 'string' },
          members: { type: 'array', items: { type: 'number' }, description: 'PR numbers, 2+' },
          rationale: { type: 'string', description: 'Why these solve the same problem at the same surface — not just same files' },
        },
      },
    },
  },
}

const VERIFY_CLOSE_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['upheld', 'reason'],
  properties: {
    upheld: { type: 'boolean', description: 'true only if the close verdict survives your attempt to refute it' },
    reason: { type: 'string' },
  },
}

const VERIFY_CLUSTER_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['upheld', 'winner', 'rationale'],
  properties: {
    upheld: { type: 'boolean' },
    winner: { type: 'number', description: 'PR number to consolidate onto; 0 when not upheld' },
    rationale: { type: 'string', description: 'Winner evidence: completeness, tests, CI, author responsiveness — and what each loser contributes' },
  },
}

function classifyPrompt(batch) {
  return (
    `READ-ONLY triage classification of these ${REPO} PRs. For each number run ` +
    `\`gh pr view <N> --repo ${REPO} --json body,files,statusCheckRollup,assignees,comments,author,createdAt,updatedAt\` ` +
    `and classify. Do NOT comment, label, close, or modify anything.\n\n` +
    `Categories:\n` +
    `- auto-close: ONLY spam, blank-template (untouched PR template AND no concrete ` +
    `problem statement anywhere — a substantive non-English description is NOT blank), ` +
    `or superseded (the same change already merged to dev — you must cite the merged ` +
    `PR/commit and confirm the code is actually on dev, similar title is not proof).\n` +
    `- recommend-close: docs-only churn, refactor-only drive-by without maintainer ` +
    `request, test-only with no linked bug, stale-abandoned (60+ days, author ` +
    `unresponsive to feedback), dirty-branch (narrow stated change bundling unrelated files).\n` +
    `- duplicate-candidate: plausibly solves the same problem as another open PR ` +
    `(check pr-overlap-check bot comments on the PR; note topics for clustering).\n` +
    `- needs-deep-review: plausible focused fix/feature worth a real review pass.\n` +
    `- keep: active discussion (author response within 14 days), assigned, green and ` +
    `awaiting review, or maintainer-authored.\n\n` +
    `NEVER classify as any close category for red CI alone. When uncertain between a ` +
    `close category and keep, choose keep with confidence=uncertain. Evidence must be ` +
    `specific and checkable.\n\nPRs: ${batch.map((p) => `#${p.number} "${p.title}" by ${p.author}`).join(', ')}`
  )
}

function clusterPrompt(rows) {
  return (
    `Group these open ${REPO} PRs into duplicate/overlap clusters. A cluster means ` +
    `2+ PRs solving the SAME problem at the SAME surface — touching the same files ` +
    `alone is not enough. Use the topics, key files, and titles below; when unsure ` +
    `about a pairing, run \`gh pr view <N> --repo ${REPO} --json body,files\` to check ` +
    `(READ-ONLY). Singletons are not clusters. Omit anything doubtful.\n\n` +
    rows
      .map((r) => `#${r.number} "${r.title}" topics=[${(r.topics || []).join(', ')}] files=[${(r.keyFiles || []).join(', ')}]`)
      .join('\n')
  )
}

function verifyClosePrompt(c) {
  return (
    `Adversarially verify this auto-close verdict on ${REPO} PR #${c.number} — your ` +
    `job is to REFUTE it. Claimed: ${c.closeReason}. Evidence: ${c.evidence}\n` +
    `Re-check from live data (READ-ONLY): for superseded, read the cited merged ` +
    `PR/commit and confirm the same behavior is on dev (git log / read the code via ` +
    `gh api). For blank-template, search body, commits, and linked issues for any ` +
    `concrete problem statement. For spam, confirm there is no plausible intent.\n` +
    `Set upheld=true ONLY if the verdict survives. If uncertain, upheld=false.`
  )
}

function verifyClusterPrompt(cluster) {
  return (
    `Adversarially verify this duplicate-cluster claim for ${REPO} PRs ` +
    `${cluster.members.map((n) => '#' + n).join(', ')} — try to REFUTE that they solve ` +
    `the same problem at the same surface. Read each PR (READ-ONLY: gh pr view --json ` +
    `body,files,statusCheckRollup,author,updatedAt).\n` +
    `If upheld, pick the consolidation winner on evidence: most complete ` +
    `implementation, best tests, green CI, responsive author; earliest PR when ` +
    `otherwise equal. In rationale, state what each non-winner contributes that the ` +
    `winner lacks (the /pr-sweep skill ports those pieces with credit). If the PRs ` +
    `are related but NOT duplicates, set upheld=false and say why.\n` +
    `Claimed rationale: ${cluster.rationale}`
  )
}

phase('Snapshot')
const scope = args && typeof args === 'object' && args.scope ? String(args.scope) : ''
const limit = args && typeof args === 'object' && Number.isInteger(args.limit) ? args.limit : 200
const snap = await agent(
  `List open PRs against dev in ${REPO} (READ-ONLY). Run: gh pr list --repo ${REPO} ` +
    `--base dev --state open --limit ${limit} --json number,title,author,isDraft,additions,deletions,changedFiles,updatedAt,assignees` +
    (scope ? ` --search ${JSON.stringify(scope)}` : '') +
    `. Compute daysSinceUpdate from updatedAt using the shell. Map author to its ` +
    `login, assigned to whether assignees is non-empty.`,
  { label: 'snapshot', phase: 'Snapshot', schema: SNAPSHOT_SCHEMA },
)
const prs = (snap && snap.prs) || []
log(`Queue snapshot: ${prs.length} open PRs${scope ? ` (scope: ${scope})` : ''}`)
if (prs.length >= limit) log(`NOTE: hit the ${limit}-PR limit — queue may be larger; pass {limit} to widen`)

phase('Classify')
const batches = []
for (let i = 0; i < prs.length; i += BATCH_SIZE) batches.push(prs.slice(i, i + BATCH_SIZE))
const batchResults = await parallel(
  batches.map((batch, i) => () =>
    agent(classifyPrompt(batch), { label: `classify#${i + 1}`, phase: 'Classify', schema: CLASSIFY_SCHEMA }),
  ),
)
const classifications = batchResults.filter(Boolean).flatMap((r) => r.classifications || [])
const byCat = {}
for (const c of classifications) byCat[c.category] = (byCat[c.category] || 0) + 1
log(`Classified ${classifications.length}/${prs.length}: ${JSON.stringify(byCat)}`)
const missed = prs.length - classifications.length
if (missed > 0) log(`NOTE: ${missed} PRs not classified (agent failures) — listed as unclassified in the plan`)

phase('Cluster')
const titleByNumber = {}
for (const p of prs) titleByNumber[p.number] = p.title
const dupRows = classifications
  .filter((c) => c.category === 'duplicate-candidate' || c.category === 'needs-deep-review')
  .map((c) => ({ number: c.number, title: titleByNumber[c.number] || '', topics: c.topics, keyFiles: c.keyFiles }))
let clusters = []
if (dupRows.length >= 2) {
  const res = await agent(clusterPrompt(dupRows), { label: 'cluster', phase: 'Cluster', schema: CLUSTER_SCHEMA })
  clusters = ((res && res.clusters) || []).filter((cl) => cl.members && cl.members.length >= 2)
}
log(`${clusters.length} duplicate cluster(s) proposed`)

phase('Verify')
const autoCloseCandidates = classifications.filter((c) => c.category === 'auto-close')
const [closeVerdicts, clusterVerdicts] = await parallel([
  () =>
    parallel(
      autoCloseCandidates.map((c) => () =>
        agent(verifyClosePrompt(c), { label: `verify-close:#${c.number}`, phase: 'Verify', schema: VERIFY_CLOSE_SCHEMA })
          .then((v) => ({ ...c, verdict: v })),
      ),
    ),
  () =>
    parallel(
      clusters.map((cl) => () =>
        agent(verifyClusterPrompt(cl), { label: `verify-cluster:${cl.topic}`, phase: 'Verify', schema: VERIFY_CLUSTER_SCHEMA })
          .then((v) => ({ ...cl, verdict: v })),
      ),
    ),
])

const upheldClose = (closeVerdicts || []).filter(Boolean).filter((c) => c.verdict && c.verdict.upheld)
const demotedClose = (closeVerdicts || []).filter(Boolean).filter((c) => !c.verdict || !c.verdict.upheld)
const upheldClusters = (clusterVerdicts || []).filter(Boolean).filter((c) => c.verdict && c.verdict.upheld)
log(`Verify: ${upheldClose.length}/${autoCloseCandidates.length} auto-closes upheld, ${upheldClusters.length}/${clusters.length} clusters upheld`)

const classifiedNumbers = new Set(classifications.map((c) => c.number))
return {
  queueSize: prs.length,
  autoClose: upheldClose.map((c) => ({ number: c.number, reason: c.closeReason, evidence: c.evidence, verifier: c.verdict.reason })),
  recommendClose: classifications
    .filter((c) => c.category === 'recommend-close')
    .concat(demotedClose.map((c) => ({ ...c, evidence: `${c.evidence} [auto-close REFUTED: ${c.verdict ? c.verdict.reason : 'verify failed'}]` })))
    .map((c) => ({ number: c.number, reason: c.closeReason, confidence: c.confidence, evidence: c.evidence })),
  clusters: upheldClusters.map((c) => ({
    topic: c.topic,
    members: c.members,
    winner: c.verdict.winner,
    rationale: c.verdict.rationale,
  })),
  deepReview: classifications.filter((c) => c.category === 'needs-deep-review').map((c) => ({ number: c.number, evidence: c.evidence })),
  keepCount: classifications.filter((c) => c.category === 'keep').length,
  unclassified: prs.map((p) => p.number).filter((n) => !classifiedNumbers.has(n)),
}
