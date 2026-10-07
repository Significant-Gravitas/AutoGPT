import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { ESLint } from "eslint";
import { format, resolveConfig } from "prettier";
import {
  ALLOWLIST,
  IMPORT_RESTRICTIONS,
  LINT_IGNORES,
  importBlocks,
  restrictionFor,
  tailwindBlocks,
} from "../eslint.design-system.mjs";

// Rebuilds eslint-allowlist.json from a lint run with the allowlist removed,
// so it lists exactly the files that break each design-system rule today.
// It is the inverse of the stale check in scripts/eslint-allowlist.test.ts,
// which imports the helpers below.
//
//   npx tsx scripts/eslint-allowlist-regenerate.ts
//
// Run it from autogpt_platform/frontend after merging branches that fix or
// delete allowlisted files. It prints what it added and removed: removals are
// the point, additions are new violations and need fixing, not allowlisting.

export const TAILWIND_RULES = [
  "no-unknown-classes",
  "no-conflicting-classes",
  "no-restricted-classes",
];

const ALLOWLIST_FILE = "eslint-allowlist.json";

// The design-system blocks without the allowlist, and without the Next.js
// presets, which only the ESLint CLI can load. The parser and plugin come
// from eslint-config-next's own dependencies.
export function eslintWithoutAllowlist() {
  const require = createRequire(import.meta.url);
  const nextRequire = createRequire(
    require.resolve("eslint-config-next/package.json"),
  );
  return new ESLint({
    cwd: process.cwd(),
    overrideConfigFile: true,
    overrideConfig: [
      { ignores: LINT_IGNORES },
      {
        files: ["**/*.{js,jsx,mjs,ts,tsx}"],
        languageOptions: {
          parser: nextRequire("@typescript-eslint/parser"),
          parserOptions: { ecmaFeatures: { jsx: true } },
        },
        plugins: {
          "@typescript-eslint": nextRequire("@typescript-eslint/eslint-plugin"),
        },
      },
      ...importBlocks({ allowlist: false }),
      ...tailwindBlocks({ allowlist: false }),
    ],
  });
}

// Each design-system violation as `imports:<restriction>:<file>` or
// `tailwind:<rule>:<file>`, the same keys the allowlist uses.
export function violationsIn(results: ESLint.LintResult[]) {
  const violations = new Set<string>();
  for (const result of results) {
    const file = result.filePath.slice(process.cwd().length + 1);
    for (const message of result.messages) {
      if (message.ruleId === "@typescript-eslint/no-restricted-imports") {
        const source = message.message.match(/^'([^']+)'/)?.[1] ?? "";
        violations.add(`imports:${restrictionFor(source)}:${file}`);
      } else if (message.ruleId?.startsWith("better-tailwindcss/")) {
        violations.add(`tailwind:${message.ruleId.split("/")[1]}:${file}`);
      }
    }
  }
  return violations;
}

function filesFor(violations: Set<string>, kind: string, key: string) {
  const prefix = `${kind}:${key}:`;
  return [...violations]
    .filter((violation) => violation.startsWith(prefix))
    .map((violation) => violation.slice(prefix.length))
    .sort();
}

function changes(before: string[], after: string[]) {
  return {
    added: after.filter((file) => !before.includes(file)),
    removed: before.filter((file) => !after.includes(file)).length,
  };
}

async function main() {
  const eslint = eslintWithoutAllowlist();
  const violations = violationsIn(await eslint.lintFiles(["."]));

  const allowlist = {
    imports: Object.fromEntries(
      Object.keys(IMPORT_RESTRICTIONS).map((key) => [
        key,
        filesFor(violations, "imports", key),
      ]),
    ),
    tailwind: Object.fromEntries(
      TAILWIND_RULES.map((rule) => [
        rule,
        filesFor(violations, "tailwind", rule),
      ]),
    ),
  };

  for (const kind of ["imports", "tailwind"] as const) {
    for (const [key, files] of Object.entries(allowlist[kind])) {
      const { added, removed } = changes(ALLOWLIST[kind][key] ?? [], files);
      console.log(
        `${kind}.${key}: ${files.length} files (${removed} removed, ${added.length} added)`,
      );
      for (const file of added) console.log(`  + ${file}`);
    }
  }

  const json = await format(JSON.stringify(allowlist), {
    ...(await resolveConfig(ALLOWLIST_FILE)),
    filepath: ALLOWLIST_FILE,
  });
  if (json === readFileSync(ALLOWLIST_FILE, "utf8")) {
    console.log(`${ALLOWLIST_FILE} is up to date.`);
    return;
  }
  writeFileSync(ALLOWLIST_FILE, json);
  console.log(`Wrote ${ALLOWLIST_FILE}.`);
}

// Run only as a script, not when the allowlist test imports the helpers.
if (resolve(process.argv[1] ?? "") === fileURLToPath(import.meta.url)) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}
