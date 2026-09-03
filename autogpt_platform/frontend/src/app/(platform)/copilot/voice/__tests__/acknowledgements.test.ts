import { describe, expect, it } from "vitest";

import { classifyUtterance, pickAcknowledgement } from "../acknowledgements";

function everyPhrase(kind: Parameters<typeof pickAcknowledgement>[0]) {
  const seen = new Set<string>();
  for (let i = 0; i < 200; i++) {
    seen.add(pickAcknowledgement(kind, null, () => i / 200));
  }
  return seen;
}

describe("pickAcknowledgement", () => {
  it("offers at least twenty distinct phrases across the registers", () => {
    const all = new Set([
      ...everyPhrase(null),
      ...everyPhrase("question"),
      ...everyPhrase("request"),
    ]);
    expect(all.size).toBeGreaterThanOrEqual(20);
  });

  it("never repeats the previous phrase", () => {
    const phrases = [...everyPhrase(null)];
    for (const previous of phrases) {
      for (let i = 0; i < 50; i++) {
        expect(pickAcknowledgement(null, previous, () => i / 50)).not.toBe(
          previous,
        );
      }
    }
  });

  it("uses a register-specific phrase once the transcript exists", () => {
    expect(everyPhrase("question")).not.toEqual(everyPhrase("request"));
  });

  it("stays register-neutral before the transcript exists", () => {
    const neutral = everyPhrase(null);
    expect(neutral.size).toBeGreaterThan(1);
    for (const phrase of neutral) {
      expect(everyPhrase("question").has(phrase)).toBe(false);
    }
  });
});

describe("classifyUtterance", () => {
  it("reads a trailing question mark", () => {
    expect(classifyUtterance("Ship it?")).toBe("question");
  });

  it("reads a question opener without punctuation", () => {
    expect(classifyUtterance("how many runs failed today")).toBe("question");
    expect(classifyUtterance("Can you check the logs")).toBe("question");
  });

  it("treats an imperative as a request", () => {
    expect(classifyUtterance("Build me a Slack agent")).toBe("request");
    expect(classifyUtterance("send that to my inbox")).toBe("request");
  });
});
