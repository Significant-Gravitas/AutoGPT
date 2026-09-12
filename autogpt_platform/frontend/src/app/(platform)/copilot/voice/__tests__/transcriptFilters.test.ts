import { describe, expect, it } from "vitest";

import { isRejectableTranscript } from "../transcriptFilters";

describe("isRejectableTranscript", () => {
  it("rejects nothing at all", () => {
    expect(isRejectableTranscript("")).toBe(true);
    expect(isRejectableTranscript("   ")).toBe(true);
    expect(isRejectableTranscript(".")).toBe(true);
  });

  it("rejects pure filler", () => {
    for (const filler of ["uh", "Um...", "hmm", "Okay.", "yeah", "uh um ah"]) {
      expect(isRejectableTranscript(filler)).toBe(true);
    }
  });

  it("rejects Whisper's silence hallucinations", () => {
    for (const phrase of [
      "Thank you.",
      "Thanks for watching!",
      "Bye.",
      "Subtitles by the Amara.org community",
    ]) {
      expect(isRejectableTranscript(phrase)).toBe(true);
    }
  });

  it("keeps a short real instruction", () => {
    expect(isRejectableTranscript("Run it")).toBe(false);
    expect(isRejectableTranscript("Stop")).toBe(false);
  });

  it("keeps an utterance that merely starts with filler", () => {
    expect(isRejectableTranscript("Um, build me a Slack agent")).toBe(false);
    expect(isRejectableTranscript("Okay, run that again")).toBe(false);
  });
});
