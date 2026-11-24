// @vitest-environment node
import path from "node:path";
import { ESLint } from "eslint";
import { describe, expect, it } from "vitest";

// A raw <img> fails lint (see "@next/next/no-img-element" in .eslintrc.json);
// a file opts out line by line with a disable that gives the reason.

const FRONTEND_ROOT = path.resolve(__dirname, "..");
const RULE = "@next/next/no-img-element";

const eslint = new ESLint({ cwd: FRONTEND_ROOT });

async function ruleMessages(source: string) {
  const [result] = await eslint.lintText(source, {
    filePath: "src/img-rule-fixture.tsx",
  });
  return result.messages.filter((message) => message.ruleId === RULE);
}

describe("the raw <img> rule", () => {
  it("fails lint on a raw <img>", async () => {
    const messages = await ruleMessages(
      'export function Probe() {\n  return <img src="/x.png" alt="x" />;\n}\n',
    );

    expect(messages).toHaveLength(1);
    expect(messages[0].severity).toBe(2);
  });

  it("passes a raw <img> whose line carries a disable", async () => {
    const messages = await ruleMessages(
      "export function Probe() {\n" +
        "  // eslint-disable-next-line @next/next/no-img-element -- a blob URL next/image cannot load\n" +
        '  return <img src="/x.png" alt="x" />;\n' +
        "}\n",
    );

    expect(messages).toHaveLength(0);
  });
});
