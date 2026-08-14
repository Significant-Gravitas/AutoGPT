import {
  buildVoicePreferences,
  type VoicePickResult,
  type VoiceSample,
} from "@/components/organisms/VoicePicker/helpers";

export type RaiseStep = "name" | "voice" | "firstJob" | "review";

export const RAISE_STEPS: RaiseStep[] = ["name", "voice", "firstJob", "review"];

export const NAME_CHIPS = ["Otto", "Nova", "Juno"];

export const VOICE_SAMPLES: VoiceSample[] = [
  {
    label: "Concise and direct",
    text: "Here's what I found and what I'd do next. No fluff — just the decision and the reason behind it.",
  },
  {
    label: "Warm and detailed",
    text: "I dug into this for you and want to walk you through what stood out, why it matters, and where I think we should head together.",
  },
];

export const RAISE_PROMPTS = {
  name: "Hi. I don't have a name yet — that's where you come in.",
  voice: (name: string) =>
    `Nice to meet you, ${name}. How should I sound when I write?`,
  firstJob:
    "What should I take on first? Pick a starter job, or skip and we'll figure it out together.",
  review: "That's me so far. Ready when you are — I'll open our first chat.",
};

export function voiceSummaryLabel(
  result: VoicePickResult,
  samples: VoiceSample[],
): string {
  if (result.choice === "custom") return "My own writing sample";
  const sample = result.choice === "a" ? samples[0] : samples[1];
  return sample?.label ?? "A voice";
}

export function resolveVoicePreferences(result: VoicePickResult): string {
  return buildVoicePreferences(result, VOICE_SAMPLES);
}
