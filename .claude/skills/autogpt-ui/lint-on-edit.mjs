#!/usr/bin/env node
// Claude Code PostToolUse hook (see .claude/settings.json): after an Edit,
// Write or MultiEdit of a .ts/.tsx file under autogpt_platform/frontend/src,
// run `npx eslint --fix` on it and hand whatever is left back to the agent.
// It never blocks: every path exits 0, including lint errors, a missing
// node_modules and a timeout.
import { spawnSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

const MARKER = "/autogpt_platform/frontend/src/";

function main() {
  const input = JSON.parse(readFileSync(0, "utf8") || "{}");
  const file = input.tool_input?.file_path ?? "";
  if (!file.includes(MARKER) || !/\.tsx?$/.test(file) || !existsSync(file)) {
    return;
  }

  const frontend = join(file.slice(0, file.indexOf(MARKER)), MARKER, "..");
  if (!existsSync(join(frontend, "node_modules"))) return;

  const result = spawnSync(
    "npx",
    ["--no-install", "eslint", "--fix", "--no-warn-ignored", file],
    { cwd: frontend, encoding: "utf8", timeout: 60_000 },
  );
  if (result.status === 0) return;

  const report = (result.stdout || result.error?.message || "").trim();
  if (!report) return;
  process.stdout.write(
    JSON.stringify({
      hookSpecificOutput: {
        hookEventName: "PostToolUse",
        additionalContext: `eslint --fix left problems in ${file}. Fix them (see autogpt_platform/frontend/DESIGN.md); do not add the file to eslint-allowlist.json.\n${report}`,
      },
    }),
  );
}

try {
  main();
} catch {
  // A broken hook must not get in the way of editing.
}
process.exit(0);
