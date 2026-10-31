// The end-of-run summary: a JSON file under results/ that names the run, the
// target, the build it was running and the UTC windows of the load, so
// perf-compare can select the steady phase in Sentry; plus a short text
// report for the terminal.
//
// handleSummary runs after teardown in its own context, so what it knows
// comes from k6's metrics, from __ENV and from one request to the target.
import http from "k6/http";

import { getEnvironmentConfig } from "../configs/environment.js";

const config = getEnvironmentConfig();

const TREND_STATS = [
  "count",
  "avg",
  "min",
  "med",
  "p(90)",
  "p(95)",
  "p(99)",
  "max",
];

export function runId() {
  return __ENV.RUN_ID || "manual";
}

function metricValue(data, name, stat) {
  const metric = data.metrics[name];
  if (!metric || !metric.values) return null;
  const value = metric.values[stat];
  return typeof value === "number" ? value : null;
}

// Trends and counters whose names carry a tag filter, such as
// http_req_duration{endpoint:store.skill_detail,phase:steady}, exist in the
// summary only when a threshold names them; thresholds.js declares one for
// every endpoint and journey, so each shows up here.
function taggedMetrics(data, baseName) {
  const out = {};
  for (const name of Object.keys(data.metrics)) {
    if (!name.startsWith(`${baseName}{`)) continue;
    const tags = name.slice(baseName.length + 1, -1);
    out[tags] = data.metrics[name];
  }
  return out;
}

function trendRow(metric) {
  const row = {};
  for (const stat of TREND_STATS) {
    if (typeof metric.values[stat] === "number") {
      row[stat] = round(metric.values[stat]);
    }
  }
  if (typeof metric.values.rate === "number")
    row.rate = round(metric.values.rate, 4);
  if (typeof metric.values.passes === "number")
    row.passes = metric.values.passes;
  if (typeof metric.values.fails === "number") row.fails = metric.values.fails;
  return row;
}

function round(value, places) {
  const factor = Math.pow(10, places === undefined ? 1 : places);
  return Math.round(value * factor) / factor;
}

function thresholdResults(data) {
  const results = {};
  let failed = 0;
  for (const name of Object.keys(data.metrics)) {
    const thresholds = data.metrics[name].thresholds;
    if (!thresholds) continue;
    for (const expr of Object.keys(thresholds)) {
      const ok = thresholds[expr].ok === true;
      if (!ok) failed++;
      results[`${name}: ${expr}`] = ok ? "pass" : "FAIL";
    }
  }
  return { results, failed };
}

// What identifies the deployed build. The backend's /health answers with its
// version and commit when it knows them; a preview also reports the commit as
// its Sentry release.
function targetBuild() {
  try {
    const res = http.get(`${config.BACKEND_URL}/health`, { timeout: "10s" });
    if (res.status !== 200) return { status: res.status };
    const body = res.json();
    return {
      status: res.status,
      commit: body.commit || body.git_sha || body.sha || null,
      version: body.version || null,
      raw: body,
    };
  } catch (e) {
    return { error: String(e) };
  }
}

// The load starts when setup() ends. With phase lengths, the steady phase
// starts after the warm-up and ends when its stage does: the window the
// thresholds judge and perf-compare should query, without the warm-up and
// without teardown's deletes. Without them the load window stands in.
function windows(data, meta) {
  const endedAt = new Date();
  const startedAt = new Date(endedAt.getTime() - data.state.testRunDurationMs);
  const setupMs = metricValue(data, "setup_duration_ms", "avg") || 0;
  const loadStartedAt = new Date(startedAt.getTime() + setupMs);
  const phases = meta.phases;
  const steadyStartedAt = phases
    ? new Date(loadStartedAt.getTime() + phases.warmupMs)
    : loadStartedAt;
  const steadyEndedAt = phases
    ? new Date(steadyStartedAt.getTime() + phases.steadyMs)
    : endedAt;
  return {
    started_at: startedAt.toISOString(),
    load_started_at: loadStartedAt.toISOString(),
    steady_started_at: steadyStartedAt.toISOString(),
    steady_ended_at: steadyEndedAt.toISOString(),
    ended_at: endedAt.toISOString(),
    setup_ms: round(setupMs, 0),
    total_ms: data.state.testRunDurationMs,
  };
}

export function buildSummary(data, meta) {
  // Before the request to the target, which takes time of its own.
  const window = windows(data, meta);
  const { results, failed } = thresholdResults(data);

  const byEndpoint = {};
  const durations = taggedMetrics(data, "http_req_duration");
  for (const tags of Object.keys(durations)) {
    const match = /endpoint:([^,}]+),phase:steady/.exec(tags);
    if (!match) continue;
    byEndpoint[match[1]] = trendRow(durations[tags]);
  }
  const failedByEndpoint = taggedMetrics(data, "http_req_failed");
  for (const tags of Object.keys(failedByEndpoint)) {
    const match = /endpoint:([^,}]+),phase:steady/.exec(tags);
    if (!match || !byEndpoint[match[1]]) continue;
    byEndpoint[match[1]].failed_rate = round(
      failedByEndpoint[tags].values.rate,
      4,
    );
  }

  const byJourney = {};
  const iterations = taggedMetrics(data, "iteration_duration");
  for (const tags of Object.keys(iterations)) {
    const match = /scenario:([^,}]+)/.exec(tags);
    if (!match) continue;
    byJourney[match[1]] = trendRow(iterations[tags]);
  }
  const dropped = taggedMetrics(data, "dropped_iterations");
  for (const tags of Object.keys(dropped)) {
    const match = /scenario:([^,}]+)/.exec(tags);
    if (!match) continue;
    byJourney[match[1]] = byJourney[match[1]] || {};
    byJourney[match[1]].dropped_iterations = dropped[tags].values.count;
  }

  const custom = {};
  for (const name of meta.customMetrics || []) {
    const steady = data.metrics[`${name}{phase:steady}`];
    const all = data.metrics[name];
    if (steady) custom[name] = trendRow(steady);
    else if (all) custom[name] = trendRow(all);
  }

  const summary = {
    run_id: runId(),
    profile: meta.profile,
    seed: meta.seed,
    target: {
      name: config.NAME,
      backend_url: config.BACKEND_URL,
      frontend_url: config.FRONTEND_URL,
      sentry: config.SENTRY || null,
      build: targetBuild(),
    },
    window,
    scenarios: meta.scenarios,
    totals: {
      http_reqs: metricValue(data, "http_reqs", "count"),
      http_req_failed_rate: metricValue(data, "http_req_failed", "rate"),
      iterations: metricValue(data, "iterations", "count"),
      checks_rate: metricValue(data, "checks", "rate"),
      dropped_iterations: metricValue(data, "dropped_iterations", "count") || 0,
      vus_max: metricValue(data, "vus_max", "max"),
    },
    http_by_endpoint: byEndpoint,
    by_journey: byJourney,
    custom_metrics: custom,
    seeding: meta.seeding ? meta.seeding(data) : null,
    coverage: meta.coverage ? meta.coverage(data) : null,
    thresholds: { failed, results },
  };
  if (meta.cost) summary.cost = meta.cost;
  return summary;
}

function pad(text, width) {
  const s = String(text);
  return s.length >= width ? s : s + " ".repeat(width - s.length);
}

function num(value) {
  return value === undefined || value === null ? "-" : String(value);
}

export function textReport(summary) {
  const lines = [];
  lines.push("");
  lines.push(
    `run ${summary.run_id}  profile ${summary.profile}  target ${summary.target.name}  build ${
      summary.target.build.commit || summary.target.build.version || "unknown"
    }`,
  );
  lines.push(
    `steady window ${summary.window.steady_started_at} / ${summary.window.steady_ended_at} (setup ${Math.round(
      summary.window.setup_ms / 1000,
    )}s, load from ${summary.window.load_started_at})`,
  );
  lines.push(
    `requests ${num(summary.totals.http_reqs)}  failed ${pct(
      summary.totals.http_req_failed_rate,
    )}  checks ${pct(summary.totals.checks_rate)}  dropped iterations ${
      summary.totals.dropped_iterations
    }  max VUs ${num(summary.totals.vus_max)}`,
  );
  lines.push("");
  lines.push(
    `${pad("endpoint (steady phase)", 36)} ${pad("count", 7)} ${pad("p50", 8)} ${pad(
      "p95",
      8,
    )} ${pad("p99", 8)} ${pad("failed", 7)}`,
  );
  for (const name of Object.keys(summary.http_by_endpoint).sort()) {
    const row = summary.http_by_endpoint[name];
    lines.push(
      `${pad(name, 36)} ${pad(num(row.count), 7)} ${pad(cell(row, "med", ms), 8)} ${pad(
        cell(row, "p(95)", ms),
        8,
      )} ${pad(cell(row, "p(99)", ms), 8)} ${pad(cell(row, "failed_rate", pct), 7)}`,
    );
  }
  if (Object.keys(summary.custom_metrics).length > 0) {
    lines.push("");
    lines.push(
      `${pad("user-facing metric (steady phase)", 36)} ${pad("count", 7)} ${pad(
        "p50",
        8,
      )} ${pad("p95", 8)} ${pad("p99", 8)} ${pad("max", 8)}`,
    );
    for (const name of Object.keys(summary.custom_metrics)) {
      const row = summary.custom_metrics[name];
      lines.push(
        `${pad(name, 36)} ${pad(num(row.count), 7)} ${pad(cell(row, "med", ms), 8)} ${pad(
          cell(row, "p(95)", ms),
          8,
        )} ${pad(cell(row, "p(99)", ms), 8)} ${pad(cell(row, "max", ms), 8)}`,
      );
    }
  }
  if (summary.seeding) {
    lines.push("");
    lines.push(`seeded: ${JSON.stringify(summary.seeding)}`);
  }
  if (summary.coverage) {
    const off = Object.keys(summary.coverage).filter(
      (name) => summary.coverage[name].status !== "covered",
    );
    lines.push("");
    if (off.length === 0) {
      lines.push("coverage: every journey ran in full");
    } else {
      lines.push(
        "NOT COVERED IN FULL: the numbers above are for what the target let the run exercise",
      );
      for (const name of off) {
        const entry = summary.coverage[name];
        const accounts = entry.accounts ? ` (${entry.accounts} accounts)` : "";
        lines.push(`  ${name}: ${entry.status}${accounts}: ${entry.reason}`);
      }
    }
  }
  lines.push("");
  const failing = Object.keys(summary.thresholds.results).filter(
    (k) => summary.thresholds.results[k] === "FAIL",
  );
  lines.push(
    `thresholds: ${Object.keys(summary.thresholds.results).length - failing.length} passed, ${
      failing.length
    } failed`,
  );
  for (const name of failing) lines.push(`  FAIL ${name}`);
  lines.push("");
  return lines.join("\n");
}

// A row with no samples shows "-" rather than the zeros k6 reports for it.
function cell(row, key, format) {
  return row.count ? format(row[key]) : "-";
}

function ms(value) {
  return value === undefined || value === null ? "-" : `${Math.round(value)}ms`;
}

function pct(rate) {
  return rate === undefined || rate === null
    ? "-"
    : `${(rate * 100).toFixed(2)}%`;
}

export function summaryOutputs(data, meta) {
  const summary = buildSummary(data, meta);
  const outputs = {
    stdout: textReport(summary),
  };
  outputs[`results/${summary.run_id}.json`] = JSON.stringify(summary, null, 2);
  return outputs;
}
