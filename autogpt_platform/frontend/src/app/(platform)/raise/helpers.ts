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
  | "budget"
  | "marketplace"
  | "skills"
  | "done";

export const STEP_ORDER: RaiseStep[] = [
  "role",
  "name",
  "color",
  "avatar",
  "about",
  "voice",
  "budget",
  "marketplace",
  "skills",
  "done",
];

export interface RaiseAttachmentDraft {
  kind: "workflow" | "skill";
  source: "marketplace" | "library";
  id: string;
  name: string;
  marketplaceKey?: string;
}

export interface RaiseKit {
  weeklyBudget: number | null;
  attachments: RaiseAttachmentDraft[];
}

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
  greeting: "Hello, I'm Autopilot. I'll help you raise your own expert.",
  roleQuestion: "First — what should your expert do for you?",
  nameQuestion: "Good pick. What do you want to call it?",
  colorQuestion: "Nice. Now choose a color for it.",
  avatarQuestion: (name: string) =>
    `Want to give ${name || "it"} a face? Upload a picture, let me generate one, or skip it.`,
  aboutQuestion: (name: string) =>
    `Anything else I should know about ${name || "your expert"}? How it should work, what matters to you — or skip it.`,
  voiceQuestion: (name: string) =>
    `How should ${name || "your expert"} sound when it writes? Pick the one that feels right.`,
  budgetQuestion: (name: string) =>
    `How much weekly budget should ${name || "your expert"} have? 500 credits is the default — pick an amount, or skip.`,
  marketplaceQuestion: (name: string) =>
    `Want ${name || "your expert"} to run workflows? Search the marketplace and your library, then add any you like — or skip.`,
  skillsQuestion: (name: string) =>
    `Should ${name || "your expert"} have extra skills? Add from your library, or a marketplace agent as a skill — or skip.`,
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
  // Outer null = not answered yet. credits null = skipped (platform default).
  budget: { credits: number | null } | null;
  marketplace: RaiseAttachmentDraft[] | null;
  skills: RaiseAttachmentDraft[] | null;
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
  budget: null,
  marketplace: null,
  skills: null,
};

const DRAFT_STORAGE_KEY = "raise-expert-draft";

export function loadDraft(): RaiseDraft {
  if (typeof window === "undefined") return EMPTY_DRAFT;
  try {
    const raw = window.sessionStorage.getItem(DRAFT_STORAGE_KEY);
    if (!raw) return EMPTY_DRAFT;
    const parsed = JSON.parse(raw) as Omit<Partial<RaiseDraft>, "step"> & {
      step?: string;
    };
    const step = parsed.step === "kit" ? "budget" : parsed.step;
    return backfillSkippedVoice({
      ...EMPTY_DRAFT,
      ...parsed,
      step: isRaiseStep(step) ? step : EMPTY_DRAFT.step,
    });
  } catch {
    return EMPTY_DRAFT;
  }
}

// A draft written by an earlier build recorded a skipped voice as a null
// label. The flow now treats null as "not answered", which would leave a
// restored session parked on the voice beat with no way forward, so a draft
// that has already moved past voice gets the sentinel back.
function backfillSkippedVoice(draft: RaiseDraft): RaiseDraft {
  if (draft.voiceLabel !== null) return draft;
  if (STEP_ORDER.indexOf(draft.step) <= STEP_ORDER.indexOf("voice")) {
    return draft;
  }
  return { ...draft, voiceLabel: VOICE_SKIPPED_LABEL };
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

function isRaiseStep(step: string | undefined): step is RaiseStep {
  return STEP_ORDER.includes(step as RaiseStep);
}

export function assembledKit(draft: RaiseDraft): RaiseKit | null {
  if (
    draft.budget === null &&
    draft.marketplace === null &&
    draft.skills === null
  ) {
    return null;
  }
  return {
    weeklyBudget: draft.budget?.credits ?? null,
    attachments: [...(draft.marketplace ?? []), ...(draft.skills ?? [])],
  };
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

export function creditsToUsdLabel(credits: number): string {
  const dollars = credits / 100;
  return Number.isInteger(dollars) ? `$${dollars}` : `$${dollars.toFixed(2)}`;
}

export function kitBudgetLabel(kit: RaiseKit | null): string | null {
  if (!kit || kit.weeklyBudget === null) return null;
  if (kit.weeklyBudget === 0) return "No weekly limit";
  return `${kit.weeklyBudget.toLocaleString()} credits (${creditsToUsdLabel(kit.weeklyBudget)}/week)`;
}

export function kitToolsLabel(kit: RaiseKit | null): string | null {
  if (!kit || kit.attachments.length === 0) return null;
  return kit.attachments.map((attachment) => attachment.name).join(", ");
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
