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

// Otto's billing words, not money in general (users talk prices all day).
// Not "credit card"; a bare "credit" needs a number or the plural.
const COST_WORDS =
  /\b(?:allowance|cents?|automation credits?|\d[\d.,]*\s*credits?|credits)\b(?!\s+cards?\b)/i;

export function mentionsCost(text: string) {
  return COST_WORDS.test(text);
}
