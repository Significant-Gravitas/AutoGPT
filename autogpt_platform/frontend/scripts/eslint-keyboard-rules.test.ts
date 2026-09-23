import { ESLint } from "eslint";
import { beforeAll, describe, expect, it } from "vitest";
import { KEY_NAMES } from "../src/lib/keyboard";

// One instance for the whole suite: resolving the Next.js + TypeScript config
// costs ~1.7s and is paid once, while each lintText call afterwards is a few
// milliseconds. Fixtures are batched per file extension for the same reason.
const eslint = new ESLint({ cwd: process.cwd() });

const EXTENSIONS = ["ts", "tsx"] as const;

function fixturePath(extension: string) {
  return `src/keyboard-rule-fixture.${extension}`;
}

async function keyboardRuleMessages(source: string, extension: string) {
  const [result] = await eslint.lintText(source, {
    filePath: fixturePath(extension),
  });
  return result.messages.filter((m) => m.ruleId === "no-restricted-syntax");
}

// The three selectors, read back off the resolved config rather than scraped
// out of .eslintrc.json, so the assertion does not depend on that file's
// textual formatting or on the directory vitest happens to run from.
async function keyboardSelectors(extension: string) {
  const config = await eslint.calculateConfigForFile(fixturePath(extension));
  const [, ...options] = config.rules?.["no-restricted-syntax"] ?? [];
  return (options as { selector: string; message: string }[]).filter((option) =>
    option.message.includes("isKey(e,"),
  );
}

describe.each(EXTENSIONS)("keyboard selectors in a .%s file", (extension) => {
  let selectors: { selector: string; message: string }[];

  beforeAll(async () => {
    selectors = await keyboardSelectors(extension);
  });

  it("is wired into the resolved ESLint config", () => {
    expect(selectors).toHaveLength(3);
  });

  it("guards exactly the KEY_NAMES list", () => {
    const lists = selectors.map((option) => {
      const match = option.selector.match(/value=\/\^\(([^)]+)\)\$\//);
      expect(
        match,
        `no key list in selector: ${option.selector}`,
      ).not.toBeNull();
      return match![1].split("|");
    });
    expect(lists).toHaveLength(3);
    for (const list of lists) {
      expect(new Set(list)).toEqual(new Set(KEY_NAMES));
    }
  });

  it("rejects every KEY_NAME in every shape the selectors cover", async () => {
    // One statement per line so a reported line number identifies the shape
    // that tripped, and every key name is proven to fire all three selectors.
    const lines = KEY_NAMES.flatMap((key) => {
      const literal = JSON.stringify(key);
      return [
        `event.key === ${literal};`,
        `${literal} === event.key;`,
        `switch (event.key) { case ${literal}: break; }`,
      ];
    });
    const source = `declare const event: { key: string };\n${lines.join("\n")}\n`;

    const messages = await keyboardRuleMessages(source, extension);
    expect(messages).toHaveLength(lines.length);
    // Every fixture line reported exactly once, none swallowed or doubled.
    expect(messages.map((m) => m.line).sort((a, b) => a - b)).toEqual(
      lines.map((_, index) => index + 2),
    );
  });

  it.each([
    'const ENTER_KEY = "Enter"; event.key === ENTER_KEY;',
    'const ENTER_KEY = "Enter"; ENTER_KEY === event.key;',
    'const ENTER_KEY = "Enter"; switch (event.key) { case ENTER_KEY: break; }',
  ])("rejects the named-constant form: %s", async (statement) => {
    const source = `declare const event: { key: string };\n${statement}\n`;
    const messages = await keyboardRuleMessages(source, extension);
    expect(messages).toHaveLength(1);
    expect(messages[0].message).toContain('use isKey(e, "Enter")');
  });

  it("leaves unrelated key handling alone", async () => {
    const source = [
      "declare const event: { key: string };",
      "declare const status: string;",
      "declare const plan: { key: string };",
      // Not a `.key` member expression.
      'status === "Enter";',
      'switch (status) { case "Enter": break; }',
      // A `.key` that is not a key name.
      'plan.key === "premium";',
      // Modifier chords and set membership are out of scope by design.
      'event.key.toLowerCase() === "k";',
      '["Enter", " "].includes(event.key);',
    ].join("\n");
    expect(await keyboardRuleMessages(source, extension)).toEqual([]);
  });
});
