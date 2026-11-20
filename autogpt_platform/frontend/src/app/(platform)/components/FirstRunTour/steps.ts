import {
  BubbleChatIcon,
  Folder01Icon,
  Rocket01Icon,
  UserGroupIcon,
  Wallet01Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import type { TourTarget } from "@/lib/tour-targets";

export interface TourContext {
  isTrial: boolean;
}

export interface TourAvailability {
  isBillingEnabled: boolean;
}

export interface TourStep {
  id: string;
  icon: IconSvgElement;
  title: string;
  /** One line. A function when the line depends on the account. */
  body: string | ((context: TourContext) => string);
  /** Where it points, then where to point when that is not on screen. */
  targets: TourTarget[];
  /** Lives in the sidebar: on a phone the menu button stands in for it. */
  inSidebar?: boolean;
  /** Per-target wording, for a step whose fallback reads differently. */
  bodyFor?: Partial<Record<TourTarget, string>>;
  /** Wording while none of the targets is on screen yet. */
  bodyWhenMissing?: string;
  /** Only shown where billing is on: a self-hosted install has no wallet. */
  requiresBilling?: boolean;
}

// One entry per step: rewording one (the allowance copy is shared with
// SECRT-3133) is an edit here, nowhere else.
export const TOUR_STEPS: TourStep[] = [
  {
    id: "chats",
    icon: BubbleChatIcon,
    title: "Every task gets its own chat",
    body: "Ask Otto for something and it starts a new chat. They all stay here, so you can pick any of them back up.",
    targets: ["chats"],
    inSidebar: true,
  },
  {
    id: "team",
    icon: UserGroupIcon,
    title: "Experts take tasks off your plate",
    body: "Otto can add experts to your team to own a task from start to finish. Team shows who's on what.",
    targets: ["team", "workspace"],
    inSidebar: true,
  },
  {
    id: "files",
    icon: Folder01Icon,
    title: "Your results are saved in Files",
    body: "Reports, drafts and anything else your team makes are kept here. Home sums up what got done.",
    targets: ["files", "workspace"],
    inSidebar: true,
  },
  {
    id: "allowance",
    icon: Wallet01Icon,
    title: "See what you have left",
    body: ({ isTrial }) =>
      isTrial
        ? "Your trial includes an allowance for chats and experts. Check what's left here, and earn extra credits."
        : "Check how much of your plan's allowance is left, and the credits your automations use.",
    // The wallet draws nothing until credits load; its footer stands in.
    targets: ["wallet", "account"],
    bodyFor: {
      account: "Your allowance and credits are here, next to your profile.",
    },
    inSidebar: true,
    requiresBilling: true,
  },
  {
    id: "start",
    icon: Rocket01Icon,
    title: "Start with one task",
    body: "Pick a suggestion and Otto gets going. Or describe your own in the box below.",
    targets: ["first-tasks", "composer"],
    bodyFor: {
      composer: "Tell Otto what you need, in your own words.",
    },
    bodyWhenMissing:
      "Otto is getting your first tasks ready. They'll show up here in a moment.",
  },
];

// Built once: downstream effects depend on the list's identity.
const STEPS_WITHOUT_BILLING = TOUR_STEPS.filter(
  (step) => !step.requiresBilling,
);

export function getAvailableSteps({ isBillingEnabled }: TourAvailability) {
  return isBillingEnabled ? TOUR_STEPS : STEPS_WITHOUT_BILLING;
}

// The list can change under an open walkthrough (billing answering late), so
// it holds a step, not a position; a dropped step hands over to the next one.
export function findStepIndex(steps: TourStep[], stepID: string | null) {
  const index = steps.findIndex((step) => step.id === stepID);
  if (stepID === null || index !== -1) return Math.max(index, 0);
  const order = TOUR_STEPS.findIndex((step) => step.id === stepID);
  const next = steps.findIndex(
    (step) => TOUR_STEPS.findIndex(({ id }) => id === step.id) > order,
  );
  return next === -1 ? steps.length - 1 : next;
}

// Cached so the same step always hands the measuring hook the same array.
const stepLists = new WeakMap<TourStep, TourStep[]>();

export function stepListOf(step: TourStep) {
  let list = stepLists.get(step);
  if (!list) {
    list = [step];
    stepLists.set(step, list);
  }
  return list;
}

export function getStepBody(
  step: TourStep,
  target: TourTarget | null,
  context: TourContext,
) {
  if (!target && step.bodyWhenMissing) return step.bodyWhenMissing;
  const override = target ? step.bodyFor?.[target] : undefined;
  if (override) return override;
  return typeof step.body === "function" ? step.body(context) : step.body;
}
