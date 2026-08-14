import type { VoiceSample } from "@/app/api/__generated__/models/voiceSample";
import {
  buildVoicePreferences,
  type VoicePickResult,
} from "@/components/organisms/VoicePicker/helpers";

export type RaiseStep = "name" | "voice" | "firstJob" | "review";

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
  namingOpener: "Let's make it official.",
  voice: (name: string) =>
    `Nice to meet you, ${name}. How should I sound when I write?`,
  firstJob:
    "What should I take on first? Pick a starter job, or skip and we'll figure it out together.",
  review: "That's me so far. Ready when you are — I'll open our first chat.",
};

export const VOICE_SKIPPED_LABEL = "I'll decide the voice later";
export const FIRST_JOB_SKIPPED_LABEL = "Skip for now";

export interface RaiseDraft {
  step: RaiseStep;
  name: string;
  voicePreferences: string;
  voiceLabel: string | null;
  firstJob: { id: string; name: string } | null;
}

export const EMPTY_DRAFT: RaiseDraft = {
  step: "name",
  name: "",
  voicePreferences: "",
  voiceLabel: null,
  firstJob: null,
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
  } catch {}
}

export function clearDraft() {
  try {
    window.sessionStorage.removeItem(DRAFT_STORAGE_KEY);
  } catch {}
}

interface RaiseMessage {
  id: string;
  role: "assistant" | "user";
  text: string;
}

const STEP_ORDER: RaiseStep[] = ["name", "voice", "firstJob", "review"];
// The naming moment reaches this flow for an AI that has already been working
// with the user — a first job would be redundant, so that step is dropped.
const STEP_ORDER_NAMING: RaiseStep[] = ["name", "voice", "review"];

function stepOrder(isNaming: boolean): RaiseStep[] {
  return isNaming ? STEP_ORDER_NAMING : STEP_ORDER;
}

export function previousStep(step: RaiseStep, isNaming = false): RaiseStep {
  const order = stepOrder(isNaming);
  const index = order.indexOf(step);
  return order[Math.max(index - 1, 0)];
}

// A draft abandoned mid-way through the regular flow can sit at the firstJob
// step or carry a picked job; naming mode has no first-job step, so both are
// dropped before the draft is reused.
export function normalizeDraftForNaming(draft: RaiseDraft): RaiseDraft {
  return {
    ...draft,
    firstJob: null,
    step: draft.step === "firstJob" ? "review" : draft.step,
  };
}

// The transcript is derived from the draft rather than accumulated, so
// back transitions and refresh restores always rebuild it consistently.
export function buildTranscript(
  draft: RaiseDraft,
  isNaming = false,
): RaiseMessage[] {
  const stepIndex = stepOrder(isNaming).indexOf(draft.step);
  const messages: RaiseMessage[] = [
    {
      id: "assistant-name",
      role: "assistant",
      text: isNaming ? RAISE_PROMPTS.namingOpener : RAISE_PROMPTS.name,
    },
  ];

  if (stepIndex >= 1) {
    messages.push(
      { id: "user-name", role: "user", text: draft.name },
      {
        id: "assistant-voice",
        role: "assistant",
        text: RAISE_PROMPTS.voice(draft.name),
      },
    );
  }
  if (stepIndex >= 2) {
    messages.push({
      id: "user-voice",
      role: "user",
      text: draft.voiceLabel ?? VOICE_SKIPPED_LABEL,
    });
    if (isNaming) {
      messages.push({
        id: "assistant-review",
        role: "assistant",
        text: RAISE_PROMPTS.review,
      });
      return messages;
    }
    messages.push({
      id: "assistant-first-job",
      role: "assistant",
      text: RAISE_PROMPTS.firstJob,
    });
  }
  if (stepIndex >= 3) {
    messages.push(
      {
        id: "user-first-job",
        role: "user",
        text: draft.firstJob?.name ?? FIRST_JOB_SKIPPED_LABEL,
      },
      {
        id: "assistant-review",
        role: "assistant",
        text: RAISE_PROMPTS.review,
      },
    );
  }
  return messages;
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
