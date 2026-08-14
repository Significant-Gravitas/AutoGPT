import type { VoiceSample } from "@/app/api/__generated__/models/voiceSample";
import { describe, expect, test } from "vitest";
import { buildVoicePreferences } from "./helpers";

const samples: VoiceSample[] = [
  { label: "Punchy and bold", text: "Stop guessing what your buyers want." },
  {
    label: "Warm and story-led",
    text: "Every campaign starts with a person, not a product.",
  },
];

describe("buildVoicePreferences", () => {
  test("preset choice a keeps the first sample as the example", () => {
    const result = buildVoicePreferences({ choice: "a" }, samples);
    expect(result).toContain("Preferred writing style: Punchy and bold.");
    expect(result).toContain("Stop guessing what your buyers want.");
    // Renders verbatim into the prompt, so it must never be JSON.
    expect(result.startsWith("{")).toBe(false);
  });

  test("preset choice b keeps the second sample as the example", () => {
    const result = buildVoicePreferences({ choice: "b" }, samples);
    expect(result).toContain("Preferred writing style: Warm and story-led.");
    expect(result).toContain("Every campaign starts with a person");
  });

  test("custom choice anchors on the user's trimmed text", () => {
    const result = buildVoicePreferences(
      { choice: "custom", customText: "  Keep it short and breezy.  " },
      samples,
    );
    expect(result).toContain("match the user's own writing sample");
    expect(result).toContain("Keep it short and breezy.");
    expect(result).not.toContain("  Keep it short and breezy.  ");
  });

  test("returns an empty string when the chosen sample is missing", () => {
    expect(buildVoicePreferences({ choice: "b" }, [samples[0]])).toBe("");
  });
});
