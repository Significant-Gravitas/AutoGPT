import type { Expert } from "@/app/api/__generated__/models/expert";
import type { VoiceSample } from "@/app/api/__generated__/models/voiceSample";
import { creditsToUsdLabel } from "@/lib/credits";

// A created expert has no samples of its own, so these two are its choices.
export const VOICE_SAMPLES: VoiceSample[] = [
  {
    label: "Concise and direct",
    text: "Here's what I found and what I'd do next. No fluff, just the decision and the reason behind it.",
  },
  {
    label: "Warm and detailed",
    text: "I dug into this for you and want to walk you through what stood out, why it matters, and where I think we should head together.",
  },
];

export function budgetLabel(expert: Expert): string {
  const budget = expert.weekly_budget;
  if (budget == null) return "Default";
  if (budget <= 0) return "No weekly limit";
  return `${creditsToUsdLabel(budget)} / week`;
}

export function voiceLabel(expert: Expert): string {
  const voice = expert.voice_preferences.trim();
  if (!voice) return "Not set yet";
  const sample = VOICE_SAMPLES.find((candidate) =>
    voice.includes(candidate.text),
  );
  if (sample) return sample.label;
  if (voice.includes("user's own writing sample")) return "Your writing sample";
  return "Custom";
}

export function countLabel(count: number, noun: string): string {
  if (count === 0) return `No ${noun}s yet`;
  return `${count} ${noun}${count === 1 ? "" : "s"}`;
}
