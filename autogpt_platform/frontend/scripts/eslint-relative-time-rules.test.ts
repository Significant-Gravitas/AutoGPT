// @vitest-environment node
import { readFileSync } from "node:fs";
import path from "node:path";
import { ESLint } from "eslint";
import ts from "typescript";
import { describe, expect, it } from "vitest";

// The formatters that read the clock may only be imported where they never
// format a time in the server's HTML or in the render that hydrates it (see
// "no-restricted-imports" in .eslintrc.json, #17309, #17310).

const FRONTEND_ROOT = path.resolve(__dirname, "..");
const RULE = "no-restricted-imports";

// One instance for the whole suite: resolving the Next.js + TypeScript config
// costs ~1.7s and is paid once.
const eslint = new ESLint({ cwd: FRONTEND_ROOT });

async function ruleMessages(source: string, filePath: string) {
  const [result] = await eslint.lintText(source, { filePath });
  return result.messages.filter((message) => message.ruleId === RULE);
}

function fixturePath(extension = ".tsx") {
  return `src/relative-time-rule-fixture${extension}`;
}

interface Override {
  files: string[];
  rules?: Record<string, unknown>;
}

// The files the rule is off for. .eslintrc.json has comments, which
// TypeScript's config-file parser reads.
function allowList() {
  const file = path.join(FRONTEND_ROOT, ".eslintrc.json");
  const { config } = ts.parseConfigFileTextToJson(
    file,
    readFileSync(file, "utf8"),
  ) as { config: { overrides: Override[] } };
  return config.overrides
    .filter((override) => override.rules?.[RULE] === "off")
    .flatMap((override) => override.files);
}

describe("the relative-time import rule", () => {
  it("rejects every way of importing a formatter that reads the clock", async () => {
    // One import per line, so a reported line names the shape that tripped.
    const lines = [
      'import { formatDistanceToNow } from "date-fns";',
      'import { formatDistanceToNowStrict } from "date-fns";',
      'import { formatDistanceToNow as ago } from "date-fns";',
      'import * as dateFns from "date-fns";',
      'import { formatDistanceToNow as one } from "date-fns/formatDistanceToNow";',
      'import { formatDistanceToNowStrict as two } from "date-fns/fp/formatDistanceToNowStrict";',
      'import { formatTimeAgo } from "@/lib/utils/time";',
      'import { formatTimeAgo as short } from "../lib/utils/time";',
      'export { formatDistanceToNow as reexported } from "date-fns";',
    ];

    const messages = await ruleMessages(`${lines.join("\n")}\n`, fixturePath());

    expect(messages.map((message) => message.line)).toEqual(
      lines.map((_, index) => index + 1),
    );
    for (const message of messages) {
      expect(message.message).toContain("RelativeTime");
    }
  });

  it("leaves the rest of date-fns and the time helpers alone", async () => {
    const source = [
      'import { format, formatDistance, formatDistanceStrict } from "date-fns";',
      'import { formatDate } from "@/lib/utils/time";',
      'import { RelativeTime } from "@/components/atoms/RelativeTime/RelativeTime";',
      'import { useRelativeTime } from "@/components/atoms/RelativeTime/useRelativeTime";',
    ].join("\n");

    expect(await ruleMessages(source, fixturePath())).toEqual([]);
  });

  it("allow-lists only files that still need it", async () => {
    const files = allowList();
    expect(files).toContain(
      "src/components/atoms/RelativeTime/useRelativeTime.ts",
    );

    for (const file of files) {
      const source = readFileSync(path.join(FRONTEND_ROOT, file), "utf8");
      // Its own text breaks the rule anywhere else, so the entry is needed...
      const elsewhere = await ruleMessages(
        source,
        fixturePath(path.extname(file)),
      );
      expect(elsewhere, file).not.toEqual([]);
      // ...and passes where it is.
      expect(await ruleMessages(source, file), file).toEqual([]);
    }
  });
});
