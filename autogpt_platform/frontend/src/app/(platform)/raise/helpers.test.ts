import type { VoiceSample } from "@/app/api/__generated__/models/voiceSample";
import { describe, expect, test } from "vitest";
import {
  getExpertLimitCode,
  raisedIdentity,
  resolveVoicePreferences,
  voiceSummaryLabel,
} from "./helpers";

const samples: VoiceSample[] = [
  { label: "Direct", text: "Do this next." },
  { label: "Warm", text: "Let's work through this together." },
];

describe("raise helpers", () => {
  test("labels preset and custom voice choices", () => {
    expect(voiceSummaryLabel({ choice: "a" }, samples)).toBe("Direct");
    expect(voiceSummaryLabel({ choice: "b" }, samples)).toBe("Warm");
    expect(
      voiceSummaryLabel(
        { choice: "custom", customText: "My own sample" },
        samples,
      ),
    ).toBe("My own writing sample");
  });

  test("resolves a preset and rejects a blank custom voice", () => {
    expect(resolveVoicePreferences({ choice: "a" }, samples)).toContain(
      "Preferred writing style: Direct.",
    );
    expect(
      resolveVoicePreferences({ choice: "custom", customText: "   " }, samples),
    ).toBeNull();
  });

  test("builds the same complete raised identity shown by the backend", () => {
    expect(raisedIdentity("Otto")).toBe(
      "I'm Otto, raised by you. I learn how you work and grow with you.",
    );
  });

  test("extracts a structured expert-limit code safely", () => {
    expect(
      getExpertLimitCode({
        detail: { code: "raised_expert_lifetime_limit", limit: 100 },
      }),
    ).toBe("raised_expert_lifetime_limit");
    expect(getExpertLimitCode({ detail: "legacy error" })).toBeNull();
    expect(getExpertLimitCode(null)).toBeNull();
  });
});
