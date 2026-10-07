import { existsSync } from "node:fs";
import { beforeAll, describe, expect, it } from "vitest";
import { ALLOWLIST, IMPORT_RESTRICTIONS } from "../eslint.design-system.mjs";
import {
  eslintWithoutAllowlist,
  TAILWIND_RULES,
  violationsIn,
} from "./eslint-allowlist-regenerate";

// The allowlist in eslint-allowlist.json is a ratchet: entries can only be
// removed. This suite fails when an entry is stale (the file was fixed or
// deleted), so fixing a file also means deleting its entry, or running
// scripts/eslint-allowlist-regenerate.ts.

const entries = [
  ...Object.entries(ALLOWLIST.imports).flatMap(([key, files]) =>
    files.map((file) => ({ kind: "imports", key, file })),
  ),
  ...Object.entries(ALLOWLIST.tailwind).flatMap(([key, files]) =>
    files.map((file) => ({ kind: "tailwind", key, file })),
  ),
];

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
    let violations = new Set<string>();

    beforeAll(async () => {
      const files = [...new Set(entries.map(({ file }) => file))].filter(
        existsSync,
      );
      violations = violationsIn(
        await eslintWithoutAllowlist().lintFiles(files),
      );
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
