import { spawnSync } from "node:child_process";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";

function lint(source: string) {
  return spawnSync(
    resolve(process.cwd(), "node_modules/.bin/eslint"),
    ["--stdin", "--stdin-filename", "src/keyboard-rule-fixture.ts"],
    {
      cwd: process.cwd(),
      encoding: "utf8",
      input: source,
    },
  );
}

function expectLintToPass(source: string) {
  const result = lint(source);
  expect(result.stderr).toBe("");
  expect(result.status).toBe(0);
}

function expectKeyboardRuleToFail(source: string) {
  const result = lint(source);
  expect(result.status).toBe(1);
  expect(result.stdout).toContain('use isKey(e, "Enter")');
  expect(result.stdout).toContain("no-restricted-syntax");
}

describe("keyboard no-restricted-syntax selectors", () => {
  it.each([
    'const status = ""; status === "Enter";',
    'const status = ""; switch (status) { case "Enter": break; }',
  ])("allows unrelated key-name literals: %s", (source) => {
    expectLintToPass(source);
  });

  it.each([
    'const event = { key: "" }; event.key === "Enter";',
    'const event = { key: "" }; const ENTER_KEY = "Enter"; event.key === ENTER_KEY;',
    'const event = { key: "" }; const ENTER_KEY = "Enter"; ENTER_KEY === event.key;',
    'const event = { key: "" }; "Enter" === event.key;',
    'const event = { key: "" }; switch (event.key) { case "Enter": break; }',
    'const event = { key: "" }; const ENTER_KEY = "Enter"; switch (event.key) { case ENTER_KEY: break; }',
  ])("rejects direct .key handling: %s", (source) => {
    expectKeyboardRuleToFail(source);
  });
});
