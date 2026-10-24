import {
  BubbleChatIcon,
  Folder01Icon,
  Rocket01Icon,
  UserGroupIcon,
  Wallet01Icon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

// The real UI the walkthrough points at. Each value is set as
// `data-tour-target` on the element it names, so a step never depends on a
// class name or a DOM path that a redesign could move.
export type TourTarget =
  | "chats"
  | "team"
  | "files"
  | "workspace"
  | "wallet"
  | "first-tasks"
  | "composer"
  | "menu";

export const TOUR_TARGET_ATTRIBUTE = "data-tour-target";

export function tourTarget(target: TourTarget) {
  return { [TOUR_TARGET_ATTRIBUTE]: target };
}

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

// The four things a new user has to learn, then where to begin. One entry per
// step and one line each: adding or rewording a step (the allowance copy is
// shared with the wallet work in SECRT-3133) is an edit here, nowhere else.
export const TOUR_STEPS: TourStep[] = [
  {
    id: "chats",
    icon: BubbleChatIcon,
    title: "Work happens in chats",
    body: "Each task gets its own chat. They all stay here, ready to pick up.",
    targets: ["chats"],
    inSidebar: true,
  },
  {
    id: "team",
    icon: UserGroupIcon,
    title: "Experts can own a task",
    body: "Your team lives here, so you can see who is working on what.",
    targets: ["team", "workspace"],
    inSidebar: true,
  },
  {
    id: "files",
    icon: Folder01Icon,
    title: "Results land in Files",
    body: "Everything your team makes is saved here. Home recaps what got done.",
    targets: ["files", "workspace"],
    inSidebar: true,
  },
  {
    id: "allowance",
    icon: Wallet01Icon,
    title: "Keep an eye on your allowance",
    body: ({ isTrial }) =>
      isTrial
        ? "Your trial includes an allowance for chats and experts. See what's left here, and earn extra credits."
        : "See how much of your plan's allowance is left here, plus your credits.",
    targets: ["wallet"],
    inSidebar: true,
    requiresBilling: true,
  },
  {
    id: "start",
    icon: Rocket01Icon,
    title: "Start here",
    body: "Pick a first task and Otto gets going on it.",
    targets: ["first-tasks", "composer"],
    bodyFor: {
      composer: "Tell Otto what you need, in your own words.",
    },
    bodyWhenMissing:
      "Otto is getting your first tasks ready. They'll show up on Home in a moment.",
  },
];

// Built once: the step list is an effect dependency downstream, so it must
// keep its identity from render to render.
const STEPS_WITHOUT_BILLING = TOUR_STEPS.filter(
  (step) => !step.requiresBilling,
);

export function getAvailableSteps({ isBillingEnabled }: TourAvailability) {
  return isBillingEnabled ? TOUR_STEPS : STEPS_WITHOUT_BILLING;
}

// One step as a list, for the hook that measures lists of targets. Cached so
// the same step always hands over the same array.
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
