import type { VoiceSample } from "@/app/api/__generated__/models/voiceSample";
import {
  buildVoicePreferences,
  type VoicePickResult,
} from "@/components/organisms/VoicePicker/helpers";

export type RaiseStep =
  | "role"
  | "name"
  | "color"
  | "avatar"
  | "about"
  | "voice"
  | "firstTask"
  | "done";

export const STEP_ORDER: RaiseStep[] = [
  "role",
  "name",
  "color",
  "avatar",
  "about",
  "voice",
  "firstTask",
  "done",
];

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
  greeting: "Hello, I'm AutoGPT. I'll help you raise your own expert.",
  roleQuestion: "First — what should your expert do for you?",
  nameQuestion: "Good pick. What do you want to call them?",
  colorQuestion: "Nice. Now choose a color for them.",
  avatarQuestion: (name: string) =>
    `Want to give ${name || "them"} a face? Upload a picture, let me generate one, or skip it.`,
  aboutQuestion:
    "Anything else I should know about them? How they should work, what matters to you — or skip it.",
  voiceQuestion: (name: string) =>
    `How should ${name || "your expert"} sound when they write? Pick the one that feels right.`,
  firstTaskQuestion: (name: string) =>
    `Last thing — what should ${name || "your expert"} start on? I'll open your first chat with it.`,
  name: "Hi. I don't have a name yet — that's where you come in.",
  voice: (name: string) =>
    `Nice to meet you, ${name}. How should I sound when I write?`,
  firstJob:
    "What should I take on first? Pick a starter job, or skip and we'll figure it out together.",
  review: "That's me so far. Ready when you are — I'll open our first chat.",
};

// Beat before each question lands, so the control that triggered it settles
// into its new state first.
export const PROMPT_DELAY_MS = 500;

export const VOICE_SKIPPED_LABEL = "I'll decide the voice later";

export interface RaiseDraft {
  step: RaiseStep;
  hasStarted: boolean;
  role: string | null;
  name: string;
  color: string | null;
  // "" once the user skips, so the question is not asked again on restore.
  avatarUrl: string | null;
  about: string | null;
  voicePreferences: string;
  voiceLabel: string | null;
  firstTask: string | null;
}

export const EMPTY_DRAFT: RaiseDraft = {
  step: "role",
  hasStarted: false,
  role: null,
  name: "",
  color: null,
  avatarUrl: null,
  about: null,
  voicePreferences: "",
  voiceLabel: null,
  firstTask: null,
};

const DRAFT_STORAGE_KEY = "raise-expert-draft";

export function loadDraft(): RaiseDraft {
  if (typeof window === "undefined") return EMPTY_DRAFT;
  try {
    const raw = window.sessionStorage.getItem(DRAFT_STORAGE_KEY);
    if (!raw) return EMPTY_DRAFT;
    const parsed = JSON.parse(raw) as Partial<RaiseDraft>;
    return { ...EMPTY_DRAFT, ...parsed };
  } catch {
    return EMPTY_DRAFT;
  }
}

export function saveDraft(draft: RaiseDraft) {
  try {
    window.sessionStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify(draft));
  } catch {
    // Draft persistence is best-effort when storage is blocked or full.
  }
}

export function clearDraft() {
  try {
    window.sessionStorage.removeItem(DRAFT_STORAGE_KEY);
  } catch {
    // Clearing is best-effort under the same storage restrictions.
  }
}

export function previousStep(step: RaiseStep): RaiseStep {
  const index = STEP_ORDER.indexOf(step);
  return STEP_ORDER[Math.max(index - 1, 0)];
}

export function voiceSummaryLabel(
  result: VoicePickResult,
  samples: VoiceSample[],
): string {
  if (result.choice === "custom") return "My own writing sample";
  const sample = result.choice === "a" ? samples[0] : samples[1];
  return sample?.label ?? "A voice";
}

export function resolveVoicePreferences(
  result: VoicePickResult,
  samples: VoiceSample[],
): string | null {
  return buildVoicePreferences(result, samples);
}

export function raisedIdentity(name: string): string {
  // Keep this preview copy aligned with backend experts_db._raised_identity.
  return `I'm ${name}, raised by you. I learn how you work and grow with you.`;
}

export function getExpertLimitCode(response: unknown): string | null {
  if (!response || typeof response !== "object" || !("detail" in response)) {
    return null;
  }
  const detail = response.detail;
  if (!detail || typeof detail !== "object" || !("code" in detail)) {
    return null;
  }
  return typeof detail.code === "string" ? detail.code : null;
}
