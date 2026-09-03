import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Linter } from "eslint";
import { describe, expect, it } from "vitest";

const configText = readFileSync(resolve(process.cwd(), ".eslintrc.json"), {
  encoding: "utf8",
});
const keyboardSelectors = [
  ...configText.matchAll(/"selector": "([^"]+)"/g),
].map(([, selector]) => ({ selector, message: "Use the keyboard helper" }));

function lint(source: string) {
  return new Linter().verify(source, {
    parserOptions: { ecmaVersion: 2022 },
    rules: {
      "no-restricted-syntax": ["error", ...keyboardSelectors],
    },
  });
}

describe("keyboard no-restricted-syntax selectors", () => {
  it.each(['status === "Enter";', 'switch (status) { case "Enter": break; }'])(
    "allows unrelated key-name literals: %s",
    (source) => {
      expect(lint(source)).toEqual([]);
    },
  );

  it.each([
    'event.key === "Enter";',
    "event.key === ENTER_KEY;",
    '"Enter" === event.key;',
    'switch (event.key) { case "Enter": break; }',
  ])("rejects direct .key handling: %s", (source) => {
    expect(lint(source)).toEqual([
      expect.objectContaining({ ruleId: "no-restricted-syntax" }),
    ]);
  });
});
