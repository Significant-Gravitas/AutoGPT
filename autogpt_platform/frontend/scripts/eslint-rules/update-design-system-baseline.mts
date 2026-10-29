/**
 * Rewrites design-system-baseline.json from the current tree:
 *
 *   pnpm lint:ds-baseline
 *
 * Runs oxlint with the design-system rules counting every violation
 * (DESIGN_SYSTEM_BASELINE=off) and stores the count per file and rule. Run it
 * after removing violations, so the lower counts are locked in, or after
 * moving a file, so its count follows the new path; reviewers see any count
 * that went up in the diff.
 */
import { spawnSync } from "node:child_process";
import { renameSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { DESIGN_SYSTEM_BASELINE_FILE } from "./design-system.mts";

const FRONTEND_ROOT = fileURLToPath(new URL("../..", import.meta.url));
const RULE_CODE = /^design-system\((.+)\)$/;

interface Diagnostic {
  code?: string;
  filename: string;
}

export function countDiagnostics(diagnostics: Diagnostic[]) {
  const counts: Record<string, Record<string, number>> = {};
  for (const { code, filename } of diagnostics) {
    const rule = RULE_CODE.exec(code ?? "")?.[1];
    if (!rule || rule === "baseline-outdated") continue;
    counts[filename] ??= {};
    counts[filename][rule] = (counts[filename][rule] ?? 0) + 1;
  }
  return Object.fromEntries(
    Object.keys(counts)
      .sort()
      .map((file) => [
        file,
        Object.fromEntries(Object.entries(counts[file]).sort()),
      ]),
  );
}

function main() {
  const result = spawnSync("pnpm", ["exec", "oxlint", "-f", "json", "src"], {
    cwd: FRONTEND_ROOT,
    encoding: "utf8",
    maxBuffer: 512 * 1024 * 1024,
    env: {
      ...process.env,
      DESIGN_SYSTEM_BASELINE: "off",
      NODE_OPTIONS: "--disable-warning=MODULE_TYPELESS_PACKAGE_JSON",
    },
  });
  if (!result.stdout.trimStart().startsWith("{")) {
    process.stderr.write(result.stderr);
    throw new Error("oxlint did not print a JSON report");
  }
  const { diagnostics } = JSON.parse(result.stdout) as {
    diagnostics: Diagnostic[];
  };
  const baseline = countDiagnostics(diagnostics);
  // Written beside the target and renamed, so lint never reads half a file.
  const partial = `${DESIGN_SYSTEM_BASELINE_FILE}.${process.pid}.tmp`;
  writeFileSync(partial, `${JSON.stringify(baseline, null, 2)}\n`);
  renameSync(partial, DESIGN_SYSTEM_BASELINE_FILE);
  const totals: Record<string, number> = {};
  for (const counts of Object.values(baseline))
    for (const [rule, count] of Object.entries(counts))
      totals[rule] = (totals[rule] ?? 0) + count;
  console.log(
    `Wrote scripts/eslint-rules/design-system-baseline.json: ${Object.keys(baseline).length} files.`,
  );
  for (const [rule, total] of Object.entries(totals).sort())
    console.log(`  design-system/${rule}: ${total}`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) main();
