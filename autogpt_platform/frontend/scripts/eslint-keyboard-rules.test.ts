import { ESLint } from "eslint";
import { describe, expect, it } from "vitest";

const eslint = new ESLint({ cwd: process.cwd() });

async function keyboardRuleMessages(source: string) {
  const [result] = await eslint.lintText(source, {
    filePath: "src/keyboard-rule-fixture.ts",
  });
  return result.messages.filter((m) => m.ruleId === "no-restricted-syntax");
}

describe("keyboard no-restricted-syntax selectors", () => {
  it.each([
    'const status = ""; status === "Enter";',
    'const status = ""; switch (status) { case "Enter": break; }',
    'const plan = { key: "" }; plan.key === "premium";',
    'const event = { key: "" }; event.key.toLowerCase() === "k";',
  ])("allows unrelated key handling: %s", async (source) => {
    expect(await keyboardRuleMessages(source)).toEqual([]);
  });

  it.each([
    'const event = { key: "" }; event.key === "Enter";',
    'const event = { key: "" }; event.key !== " ";',
    'const event = { key: "" }; const ENTER_KEY = "Enter"; event.key === ENTER_KEY;',
    'const event = { key: "" }; const ENTER_KEY = "Enter"; ENTER_KEY === event.key;',
    'const event = { key: "" }; "Enter" === event.key;',
    'const event = { key: "" }; switch (event.key) { case "Enter": break; }',
    'const event = { key: "" }; const ENTER_KEY = "Enter"; switch (event.key) { case ENTER_KEY: break; }',
  ])("rejects direct .key handling: %s", async (source) => {
    const messages = await keyboardRuleMessages(source);
    expect(messages).toHaveLength(1);
    expect(messages[0].message).toContain('use isKey(e, "Enter")');
  });
});
