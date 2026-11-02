/**
 * Prints the medians of two labelled sets of `pnpm perf` runs side by side,
 * one markdown table per scenario, ready to paste into a PR. With only --a,
 * prints that set's medians, spread and rankings (used for the baseline).
 *
 *   pnpm perf:compare --a before --b after [--scenario copilot-stream-reply]
 *   pnpm perf:compare --a baseline
 */
import fs from "fs";
import { parseArgs } from "util";
import { PERF_OUTPUT_DIR, scenarioOutputDir } from "./lib/paths";
import {
  formatComparison,
  formatSummary,
  loadRuns,
  safeLabel,
} from "./lib/report";

function main() {
  const { values } = parseArgs({
    options: {
      a: { type: "string" },
      b: { type: "string" },
      scenario: { type: "string" },
    },
  });
  if (!values.a) throw new Error("Pass --a <label> [--b <label>].");
  const a = safeLabel(values.a);
  const b = values.b ? safeLabel(values.b) : null;
  const scenarios = values.scenario
    ? [values.scenario]
    : fs.existsSync(PERF_OUTPUT_DIR)
      ? fs.readdirSync(PERF_OUTPUT_DIR).sort()
      : [];

  const sections = scenarios.flatMap((scenario) => {
    const dir = scenarioOutputDir(scenario);
    const runsA = loadRuns(dir, a);
    if (runsA.length === 0) return [];
    if (!b) return [formatSummary({ label: a, runs: runsA })];
    const runsB = loadRuns(dir, b);
    if (runsB.length === 0) return [];
    return [
      formatComparison({
        scenario,
        a: { label: a, runs: runsA },
        b: { label: b, runs: runsB },
      }),
    ];
  });
  if (sections.length === 0) {
    const labels = b ? `both "${a}" and "${b}"` : `"${a}"`;
    throw new Error(
      `No scenario in ${PERF_OUTPUT_DIR} has runs labelled ${labels}.`,
    );
  }
  console.log(sections.join("\n\n"));
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : error);
  process.exit(1);
}
