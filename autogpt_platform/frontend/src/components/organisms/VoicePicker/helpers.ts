import type { VoiceSample } from "@/app/api/__generated__/models/voiceSample";

export type VoicePickChoice = "a" | "b" | "custom";

export interface VoicePickResult {
  choice: VoicePickChoice;
  customText?: string;
}

// The result maps to the string stored in voice_preferences, which renders
// verbatim into the prompt's <voice_preferences> block — so it must read as
// plain guidance, never JSON. A preset pick keeps the chosen sample as a
// concrete example; a custom pick anchors on the user's own words.
export function buildVoicePreferences(
  result: VoicePickResult,
  samples: VoiceSample[],
): string {
  if (result.choice === "custom") {
    const text = (result.customText ?? "").trim();
    return `Preferred writing style: match the user's own writing sample below.\n\nExample to match:\n\n${text}`;
  }
  const sample = result.choice === "a" ? samples[0] : samples[1];
  if (!sample) return "";
  return `Preferred writing style: ${sample.label}.\n\nExample to match:\n\n${sample.text}`;
}
