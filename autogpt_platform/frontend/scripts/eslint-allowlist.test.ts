import { existsSync } from "node:fs";
import { createRequire } from "node:module";
import { ESLint } from "eslint";
import { beforeAll, describe, expect, it } from "vitest";
import {
  ALLOWLIST,
  IMPORT_RESTRICTIONS,
  importBlocks,
  restrictionFor,
  tailwindBlocks,
} from "../eslint.design-system.mjs";

// The allowlist in eslint-allowlist.json is a ratchet: entries can only be
// removed. This suite fails when an entry is stale (the file was fixed or
// deleted), so fixing a file also means deleting its entry.

const TAILWIND_RULES = ["no-unknown-classes", "no-conflicting-classes"];

const entries = [
  ...Object.entries(ALLOWLIST.imports).flatMap(([key, files]) =>
    files.map((file) => ({ kind: "imports", key, file })),
  ),
  ...Object.entries(ALLOWLIST.tailwind).flatMap(([key, files]) =>
    files.map((file) => ({ kind: "tailwind", key, file })),
  ),
];

// The design-system blocks without the allowlist, and without the Next.js
// presets, which only the ESLint CLI can load. The parser and plugin come
// from eslint-config-next's own dependencies.
function eslintWithoutAllowlist() {
  const require = createRequire(import.meta.url);
  const nextRequire = createRequire(
    require.resolve("eslint-config-next/package.json"),
  );
  return new ESLint({
    cwd: process.cwd(),
    overrideConfigFile: true,
    overrideConfig: [
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

describe("eslint-allowlist.json", () => {
  it("names only known restrictions and rules", () => {
    expect(
      Object.keys(ALLOWLIST.imports).filter(
        (key) => !(key in IMPORT_RESTRICTIONS),
      ),
    ).toEqual([]);
    expect(
      Object.keys(ALLOWLIST.tailwind).filter(
        (key) => !TAILWIND_RULES.includes(key),
      ),
    ).toEqual([]);
  });

  it("keeps every list sorted and free of duplicates", () => {
    for (const files of [
      ...Object.values(ALLOWLIST.imports),
      ...Object.values(ALLOWLIST.tailwind),
    ]) {
      expect(files).toEqual([...new Set(files)].sort());
    }
  });

  it("lists only files that exist", () => {
    expect(entries.filter(({ file }) => !existsSync(file))).toEqual([]);
  });

  describe("has no stale entries", () => {
    const violations = new Set<string>();

    beforeAll(async () => {
      const files = [...new Set(entries.map(({ file }) => file))].filter(
        existsSync,
      );
      const results = await eslintWithoutAllowlist().lintFiles(files);
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
    }, 120_000);

    it("every entry still has the violation it excuses", () => {
      const stale = entries
        .filter(({ kind, key, file }) => {
          return !violations.has(`${kind}:${key}:${file}`);
        })
        .map(({ kind, key, file }) => `${kind}.${key}: ${file}`);
      expect(
        stale,
        "Delete these entries from eslint-allowlist.json: the files no longer break the rule.",
      ).toEqual([]);
    });
  });
});
