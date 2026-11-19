import { Idea01Icon } from "@hugeicons/core-free-icons";
import type { TourStep } from "../FirstRunTour/steps";
import { TIPS, type TipID } from "./tips";

// Local cache of each tip's step (keyed by user id), in case the write is
// slow or fails.
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

export function isFirstSecondChat(mostSeen: number | null, count: number) {
  return mostSeen !== null && mostSeen < 2 && count >= 2;
}

// Built once per tip: the measuring hook keys on the list's identity.
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
