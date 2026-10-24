import { Idea01Icon } from "@hugeicons/core-free-icons";
import type { TourStep } from "../FirstRunTour/steps";
import { TIPS, type TipID } from "./tips";

// Local cache of each tip's onboarding step, so a dismissed tip stays gone
// before the write lands and if it fails. The value is the user id: another
// account on the same browser still gets its own tips.
const SEEN_KEY_PREFIX = "autogpt:contextual-tip-seen:";

export function peekTipSeen(id: TipID, userID: string | null | undefined) {
  if (typeof window === "undefined" || !userID) return false;
  try {
    return window.localStorage.getItem(SEEN_KEY_PREFIX + id) === userID;
  } catch {
    return false;
  }
}

export function setTipSeen(id: TipID, userID: string | null | undefined) {
  if (typeof window === "undefined" || !userID) return;
  try {
    window.localStorage.setItem(SEEN_KEY_PREFIX + id, userID);
  } catch {
    // The server step still records it.
  }
}

// The chat list just went from one chat to two: the first second thread.
export function isFirstSecondChat(mostSeen: number | null, count: number) {
  return mostSeen === 1 && count >= 2;
}

// A tip measured like a walkthrough step, so it follows its target the same
// way. Built once per tip: the measuring hook keys on the list's identity.
const tipSteps = new Map<TipID, TourStep[]>();

export function tipStepOf(id: TipID) {
  let steps = tipSteps.get(id);
  if (!steps) {
    const tip = TIPS[id];
    steps = [
      {
        id: `tip-${id}`,
        icon: Idea01Icon,
        title: tip.title,
        body: "",
        targets: tip.targets,
      },
    ];
    tipSteps.set(id, steps);
  }
  return steps;
}
