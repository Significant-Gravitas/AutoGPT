import { describe, expect, it } from "vitest";

import { pickAcknowledgement } from "../acknowledgements";

function everyPhrase(previous: string | null = null) {
  const seen = new Set<string>();
  for (let i = 0; i < 200; i++) {
    seen.add(pickAcknowledgement(previous, () => i / 200));
  }
  return seen;
}

describe("pickAcknowledgement", () => {
  it("offers at least twenty phrases, so a session does not sound canned", () => {
    expect(everyPhrase().size).toBeGreaterThanOrEqual(20);
  });

  it("never repeats the previous phrase", () => {
    for (const previous of everyPhrase()) {
      expect(everyPhrase(previous).has(previous)).toBe(false);
    }
  });

  it("offers nothing that only fits an instruction or only fits a question", () => {
    // It is chosen before the transcript exists, so every phrase has to sit
    // equally well after "what did I run yesterday" and "build me an agent".
    for (const phrase of everyPhrase()) {
      expect(phrase).not.toMatch(/\b(doing|get that|take care of|started)\b/i);
    }
  });
});
