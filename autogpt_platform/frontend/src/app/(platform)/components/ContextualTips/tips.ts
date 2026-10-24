import type { PostV1CompleteOnboardingStepStep } from "@/app/api/__generated__/models/postV1CompleteOnboardingStepStep";
import type { TourTarget } from "../FirstRunTour/steps";

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

// One-line tips for the first time each surface matters, instead of
// front-loading them into the walkthrough.
export const TIPS: Record<TipID, Tip> = {
  "second-chat": {
    step: "CHATS_TIP",
    targets: ["chats", "menu"],
    title: "You have two chats now",
    body: "Each task gets its own chat. Switch between them here.",
  },
  "first-cost": {
    step: "COST_TIP",
    targets: ["wallet", "menu"],
    requiresBilling: true,
    title: "Where costs come from",
    body: ({ isTrial }) =>
      isTrial
        ? "Chats and experts use your trial allowance, and automations use credits. Check both here."
        : "Chats and experts use your plan's allowance, and automations use credits. Check both here.",
  },
  "first-expert": {
    step: "EXPERT_TIP",
    targets: ["team", "workspace", "menu"],
    title: "Experts can own tasks",
    body: "Once you approve, this expert joins your team and can take tasks off your plate. Team shows who owns what.",
  },
};

export function getTipBody(id: TipID, context: TipContext) {
  const { body } = TIPS[id];
  return typeof body === "function" ? body(context) : body;
}

// Otto's own billing words, not money in general: a small-business owner
// talks about prices and costs all day, and none of that is about the trial.
// "credit card" is the card on file, not the credit balance.
// "credit" alone is too common ("give credit to"), so it needs a number or
// the plural: "2 credits", "1 credit", "your credits".
const COST_WORDS =
  /\b(?:allowance|cents?|automation credits?|\d[\d.,]*\s*credits?|credits)\b(?!\s+cards?\b)/i;

export function mentionsCost(text: string) {
  return COST_WORDS.test(text);
}
