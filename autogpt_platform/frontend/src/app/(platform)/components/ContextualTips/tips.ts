import type { PostV1CompleteOnboardingStepStep } from "@/app/api/__generated__/models/postV1CompleteOnboardingStepStep";
import type { TourTarget } from "@/lib/tour-targets";

export type TipID = "second-chat" | "first-cost" | "first-expert";

export interface TipContext {
  isTrial: boolean;
}

interface Tip {
  /** Records the tip as seen, once per user across devices. */
  step: PostV1CompleteOnboardingStepStep;
  /** Where it points, then where to point when that is not on screen. */
  targets: TourTarget[];
  title: string;
  body: string | ((context: TipContext) => string);
  /** Only where billing is on: a self-hosted install has no wallet. */
  requiresBilling?: boolean;
}

export const TIPS: Record<TipID, Tip> = {
  "second-chat": {
    step: "CHATS_TIP",
    targets: ["chats", "menu"],
    title: "You have two chats now",
    body: "Each task keeps its own chat. Switch between them here.",
  },
  "first-cost": {
    step: "COST_TIP",
    targets: ["wallet", "menu"],
    requiresBilling: true,
    title: "Where costs come from",
    body: ({ isTrial }) =>
      isTrial
        ? "Chats and experts use your trial allowance. Automations use credits. You can check both here."
        : "Chats and experts use your plan's allowance. Automations use credits. You can check both here.",
  },
  "first-expert": {
    step: "EXPERT_TIP",
    targets: ["team", "workspace", "menu"],
    title: "Experts can own tasks",
    body: "Approve this expert and they join your team to take it from here. Team shows who owns what.",
  },
};

export function getTipBody(id: TipID, context: TipContext) {
  const { body } = TIPS[id];
  return typeof body === "function" ? body(context) : body;
}

// Units of Otto's own work that a cost in cents is quoted against.
const WORK_UNIT =
  "(?:image|photo|picture|run|message|reply|task|search|call|video|page|step|request|generation)s?";

// Otto's billing words in the sense the product uses them, not money in
// general: people talk tax credits, share prices and baggage allowances all
// day, and a tip spent on one of those is gone for the real moment.
const COST_PHRASES = [
  // The person's allowance, or the trial's or plan's: not a baggage one.
  /\byour\s+(?:(?:trial|plan|usage|remaining|daily|weekly|monthly)\s+)?allowance\b/i,
  /\b(?:trial|plan|usage)\s+allowance\b/i,
  // Credits as the balance: a count of them, ours by name, or what is left.
  /\b\d[\d.,]*\s*credits?\b(?!\s+cards?\b)/i,
  /\bautomation\s+credits?\b/i,
  /\byour\s+(?:remaining\s+|automation\s+)?credits\b/i,
  /\bcredits?\s+(?:left|remaining)\b/i,
  // Cents against a unit of Otto's work: "a few cents per image".
  new RegExp(
    `\\bcents?\\b[^.!?\\n]{0,25}\\b(?:per|each|every|for each|a|an)\\s+${WORK_UNIT}\\b`,
    "i",
  ),
  new RegExp(
    `\\b(?:per|each|every)\\s+${WORK_UNIT}\\b[^.!?\\n]{0,25}\\bcents?\\b`,
    "i",
  ),
];

export function mentionsCost(text: string) {
  return COST_PHRASES.some((phrase) => phrase.test(text));
}
