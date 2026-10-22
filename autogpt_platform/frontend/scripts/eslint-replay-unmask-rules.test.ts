import { ESLint } from "eslint";
import { describe, expect, it } from "vitest";

// One instance for the whole suite: resolving the Next.js + TypeScript config
// is paid once, each lintText call afterwards is a few milliseconds.
const eslint = new ESLint({ cwd: process.cwd() });

const HEADER = `
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { UNMASKED_IN_REPLAY, fixedCopy, type FixedCopy } from "@/lib/replay-privacy";
import { cn } from "@/lib/utils";
declare const user: { name: string };
declare const label: FixedCopy;
declare const open: boolean;
declare const props: Record<string, string>;
declare const PinIcon: never;
`;

async function replayRuleMessages(jsx: string) {
  const source = `${HEADER}\nexport function Fixture() {\n  return (\n    <>${jsx}</>\n  );\n}\n`;
  const [result] = await eslint.lintText(source, {
    filePath: "src/replay-unmask-rule-fixture.tsx",
  });
  return result.messages.filter(
    (message) =>
      message.ruleId === "no-restricted-syntax" &&
      /unmasked in session replays|raw sentry-unmask/.test(message.message),
  );
}

describe("ESLint guard on what an unmasked element may show", () => {
  it.each([
    ["a literal child", `<Text variant="body" unmask>Workflows</Text>`],
    ["a string expression", `<Text variant="body" unmask>{"Don't"}</Text>`],
    [
      "a choice between literals",
      `<span className={UNMASKED_IN_REPLAY}>{open ? "Unpin chat" : "Pin chat"}</span>`,
    ],
    [
      "fixedCopy(...)",
      `<span className={cn("truncate", UNMASKED_IN_REPLAY)}>{fixedCopy(label)}</span>`,
    ],
    [
      "an icon and a literal aria-label",
      `<Button unmask aria-label="Add files"><Icon icon={PinIcon} /></Button>`,
    ],
    [
      "a masked element with anything",
      `<Text variant="body">{user.name}</Text>`,
    ],
  ])("allows %s", async (_name, jsx) => {
    expect(await replayRuleMessages(jsx)).toEqual([]);
  });

  it.each([
    ["a runtime value", `<Text variant="body" unmask>{user.name}</Text>`],
    [
      "a template literal",
      `<Text variant="body" unmask>{\`Hi \${user.name}\`}</Text>`,
    ],
    [
      "interpolation through the class",
      `<span className={cn("truncate", UNMASKED_IN_REPLAY)}>{label}</span>`,
    ],
    [
      "a nested element",
      `<Text variant="body" unmask>Hi <b>{user.name}</b></Text>`,
    ],
    [
      "a dynamic aria-label",
      `<Button unmask aria-label={\`Delete \${user.name}\`}>Delete</Button>`,
    ],
    [
      "a dynamic data-text",
      `<span className={UNMASKED_IN_REPLAY} data-text={user.name}>Review</span>`,
    ],
    ["spread props", `<Button unmask {...props}>Save</Button>`],
    [
      "a raw class",
      `<span className="truncate sentry-unmask">{user.name}</span>`,
    ],
    ["a raw attribute", `<span data-sentry-unmask>{user.name}</span>`],
  ])("rejects %s", async (_name, jsx) => {
    expect(await replayRuleMessages(jsx)).not.toEqual([]);
  });
});
