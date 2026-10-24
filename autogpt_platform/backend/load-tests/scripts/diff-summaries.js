#!/usr/bin/env node
// Puts two run summaries (results/<RUN_ID>.json) side by side: per endpoint
// and per user-facing metric, the steady-phase p50 and p95 of each run and
// the change. It is a quick client-side read of two runs, not a verdict: the
// statistics that decide faster or slower are perf-compare's, in Sentry.
//
//   node scripts/diff-summaries.js results/<control>.json results/<candidate>.json
import fs from "node:fs";

const [controlPath, candidatePath] = process.argv.slice(2);
if (!controlPath || !candidatePath) {
  console.error(
    "usage: node scripts/diff-summaries.js <control summary.json> <candidate summary.json>",
  );
  process.exit(2);
}

const control = JSON.parse(fs.readFileSync(controlPath, "utf8"));
const candidate = JSON.parse(fs.readFileSync(candidatePath, "utf8"));

function pad(text, width, right) {
  const s = String(text);
  if (s.length >= width) return s;
  return right
    ? " ".repeat(width - s.length) + s
    : s + " ".repeat(width - s.length);
}

function ms(value) {
  return value === undefined || value === null ? "-" : `${Math.round(value)}`;
}

function change(a, b) {
  if (!a || !b) return "-";
  const rel = ((b - a) / a) * 100;
  return `${rel >= 0 ? "+" : ""}${rel.toFixed(0)}%`;
}

function table(title, left, right) {
  const names = Array.from(
    new Set(Object.keys(left || {}).concat(Object.keys(right || {}))),
  ).sort();
  if (names.length === 0) return;
  console.log("");
  console.log(title);
  console.log(
    `${pad("", 34)} ${pad("n", 11, true)} ${pad("p50 ctl", 8, true)} ${pad(
      "p50 cand",
      9,
      true,
    )} ${pad("chg", 6, true)} ${pad("p95 ctl", 8, true)} ${pad("p95 cand", 9, true)} ${pad(
      "chg",
      6,
      true,
    )}`,
  );
  for (const name of names) {
    const a = (left || {})[name] || {};
    const b = (right || {})[name] || {};
    if (!a.count && !b.count) continue;
    console.log(
      `${pad(name, 34)} ${pad(`${a.count || 0}/${b.count || 0}`, 11, true)} ${pad(
        ms(a.med),
        8,
        true,
      )} ${pad(ms(b.med), 9, true)} ${pad(change(a.med, b.med), 6, true)} ${pad(
        ms(a["p(95)"]),
        8,
        true,
      )} ${pad(ms(b["p(95)"]), 9, true)} ${pad(change(a["p(95)"], b["p(95)"]), 6, true)}`,
    );
  }
}

// Rate metrics (copilot_turn_errors, graph_run_errors, ws_errors) carry a
// rate and the samples behind it (passes and fails), not a count of
// timings, so they get a table of their own: the rate on each side, its
// samples, and the change in percentage points. A counter's rate is per
// second, and it has no passes or fails.
function rateTable(title, left, right) {
  const isRate = (row) =>
    Boolean(row) &&
    typeof row.rate === "number" &&
    (typeof row.passes === "number" || typeof row.fails === "number");
  const names = Array.from(
    new Set(Object.keys(left || {}).concat(Object.keys(right || {}))),
  )
    .filter((name) => isRate((left || {})[name]) || isRate((right || {})[name]))
    .sort();
  if (names.length === 0) return;
  const percent = (row) =>
    isRate(row) ? `${(row.rate * 100).toFixed(2)}%` : "-";
  const samples = (row) =>
    isRate(row) ? (row.passes || 0) + (row.fails || 0) : 0;
  console.log("");
  console.log(title);
  console.log(
    `${pad("", 34)} ${pad("n", 11, true)} ${pad("ctl", 8, true)} ${pad(
      "cand",
      9,
      true,
    )} ${pad("chg", 8, true)}`,
  );
  for (const name of names) {
    const a = (left || {})[name];
    const b = (right || {})[name];
    const points =
      isRate(a) && isRate(b)
        ? `${b.rate >= a.rate ? "+" : ""}${((b.rate - a.rate) * 100).toFixed(2)}pp`
        : "-";
    console.log(
      `${pad(name, 34)} ${pad(`${samples(a)}/${samples(b)}`, 11, true)} ${pad(
        percent(a),
        8,
        true,
      )} ${pad(percent(b), 9, true)} ${pad(points, 8, true)}`,
    );
  }
}

// The steady phase; summaries from before it was recorded have the load.
function steadyWindow(summary) {
  const w = summary.window;
  return `${w.steady_started_at || w.load_started_at} / ${w.steady_ended_at || w.ended_at}`;
}

function coverageLine(summary) {
  const coverage = summary.coverage || {};
  return Object.keys(coverage)
    .sort()
    .map((name) => `${name}=${coverage[name].status}`)
    .join(" ");
}

console.log(
  `control   ${control.run_id}  ${control.target.name}  profile ${control.profile}  ${steadyWindow(control)}`,
);
console.log(
  `candidate ${candidate.run_id}  ${candidate.target.name}  profile ${candidate.profile}  ${steadyWindow(candidate)}`,
);
if (control.profile !== candidate.profile || control.seed !== candidate.seed) {
  console.log(
    "note: the runs differ in profile or seed, so they did not send the same load",
  );
}
if (coverageLine(control) !== coverageLine(candidate)) {
  console.log(
    `note: the runs covered different journeys, so they did not send the same load:\n  control   ${coverageLine(
      control,
    )}\n  candidate ${coverageLine(candidate)}`,
  );
}
console.log(
  `requests ${control.totals.http_reqs} / ${candidate.totals.http_reqs}; failed ${(
    control.totals.http_req_failed_rate * 100
  ).toFixed(
    2,
  )}% / ${(candidate.totals.http_req_failed_rate * 100).toFixed(2)}%; dropped iterations ${
    control.totals.dropped_iterations
  } / ${candidate.totals.dropped_iterations}`,
);

table(
  "HTTP by endpoint (ms, steady phase)",
  control.http_by_endpoint,
  candidate.http_by_endpoint,
);
table(
  "User-facing metrics (ms, steady phase)",
  control.custom_metrics,
  candidate.custom_metrics,
);
rateTable(
  "Failure rates (steady phase)",
  control.custom_metrics,
  candidate.custom_metrics,
);

console.log("");
console.log(
  "Verdicts come from perf-compare on the Sentry traces of these windows; see README.md.",
);
