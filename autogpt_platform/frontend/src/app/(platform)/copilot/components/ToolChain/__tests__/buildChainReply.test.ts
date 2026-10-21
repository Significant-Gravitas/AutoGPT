import { describe, expect, it } from "vitest";
import { buildChainReply, type ChainActionEntry } from "../chainActions";

function entry(
  message: string | null,
  overrides: Partial<ChainActionEntry> = {},
): ChainActionEntry {
  return {
    id: message ?? "none",
    ready: true,
    buildMessage: () => message,
    ...overrides,
  };
}

const RERUN =
  "I've configured the required credentials. Please re-run this step now.";
const CONTINUE =
  "I've configured the required credentials. I've connected my account. Please continue.";

describe("buildChainReply", () => {
  it("confirms credentials once when several cards ask for the same account", () => {
    const reply = buildChainReply([
      entry(RERUN, { credentialsOnly: true }),
      entry(CONTINUE, { credentialsOnly: true }),
    ]);

    expect(reply).toBe(RERUN);
  });

  it("keeps every line that carries more than a confirmation", () => {
    const reply = buildChainReply([
      entry(RERUN, { credentialsOnly: true }),
      entry('Run with these inputs: {"repo_url": "x"}'),
      entry("Here are my answers: blue"),
      entry(null),
    ]);

    expect(reply).toBe(
      [
        RERUN,
        'Run with these inputs: {"repo_url": "x"}',
        "Here are my answers: blue",
      ].join("\n\n"),
    );
  });
});
