import { TOUR_TARGET_ATTRIBUTE, type TourStep, type TourTarget } from "./steps";

// The onboarding steps that mean "this person has had the first-run intro".
// CAPABILITY_CARDS is the old four-card deck: someone who already sat through
// it is an existing user, and the walkthrough does not open on them by itself
// (they can still replay it from the profile menu).
export const TOUR_DONE_STEPS = ["FIRST_RUN_TOUR", "CAPABILITY_CARDS"];

// Local cache of FIRST_RUN_TOUR, so the walkthrough stays closed the instant
// it ends rather than after the round trip, and stays closed if that request
// failed. The value is the user id: another account on the same browser
// still gets its own first run.
const SEEN_KEY = "autogpt:first-run-tour-seen";

export function peekFirstRunTourSeen(userID: string | null | undefined) {
  if (typeof window === "undefined" || !userID) return false;
  try {
    return window.localStorage.getItem(SEEN_KEY) === userID;
  } catch {
    return false;
  }
}

export function setFirstRunTourSeen(userID: string | null | undefined) {
  if (typeof window === "undefined" || !userID) return;
  try {
    window.localStorage.setItem(SEEN_KEY, userID);
  } catch {
    // Private mode or a full quota: the server step still records it.
  }
}

export interface Box {
  top: number;
  left: number;
  width: number;
  height: number;
}

// A target only counts when it is laid out and on screen: the collapsed
// sidebar keeps the full nav in the DOM with `display: none`, and the phone
// keeps it in a closed sheet.
export function findTourTarget(target: TourTarget) {
  const nodes = document.querySelectorAll<HTMLElement>(
    `[${TOUR_TARGET_ATTRIBUTE}="${target}"]`,
  );
  for (const node of Array.from(nodes)) {
    const rect = node.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) continue;
    if (rect.bottom <= 0 || rect.right <= 0) continue;
    if (rect.top >= window.innerHeight || rect.left >= window.innerWidth)
      continue;
    return node;
  }
  return null;
}

export interface ResolvedStop {
  target: TourTarget | null;
  element: HTMLElement | null;
  /** True when a phone's menu button stands in for a sidebar target. */
  viaMenu: boolean;
}

export function resolveStop(step: TourStep): ResolvedStop {
  for (const target of step.targets) {
    const element = findTourTarget(target);
    if (element) return { target, element, viaMenu: false };
  }
  if (step.inSidebar) {
    const menu = findTourTarget("menu");
    if (menu) return { target: "menu", element: menu, viaMenu: true };
  }
  return { target: null, element: null, viaMenu: false };
}

const SPOTLIGHT_PADDING = 6;
// A long chat list would light up the whole sidebar; the header and the first
// rows are enough to say "here". Everything else is shown whole.
const CAPPED_TARGETS: Partial<Record<TourTarget, number>> = { chats: 220 };

// Whether the part of a target the spotlight lights up runs past the edge
// of the screen. The walkthrough locks page scrolling, so it scrolls such a
// target in itself rather than leave a sliver of it lit.
export function isPartlyOffScreen(
  element: HTMLElement,
  target: TourTarget | null,
) {
  const cap = (target && CAPPED_TARGETS[target]) ?? Number.POSITIVE_INFINITY;
  const rect = element.getBoundingClientRect();
  const bottom = Math.min(rect.bottom, rect.top + cap);
  return (
    rect.top < 0 ||
    rect.left < 0 ||
    bottom > window.innerHeight ||
    rect.right > window.innerWidth
  );
}

export function getSpotlightBox(
  element: HTMLElement,
  target: TourTarget | null,
): Box {
  const cap = (target && CAPPED_TARGETS[target]) ?? Number.POSITIVE_INFINITY;
  const rect = element.getBoundingClientRect();
  const top = Math.max(rect.top - SPOTLIGHT_PADDING, SPOTLIGHT_PADDING);
  const left = Math.max(rect.left - SPOTLIGHT_PADDING, SPOTLIGHT_PADDING);
  const right = Math.min(
    rect.right + SPOTLIGHT_PADDING,
    window.innerWidth - SPOTLIGHT_PADDING,
  );
  const bottom = Math.min(
    rect.bottom + SPOTLIGHT_PADDING,
    window.innerHeight - SPOTLIGHT_PADDING,
    top + cap,
  );
  return {
    top,
    left,
    width: Math.max(right - left, 0),
    height: Math.max(bottom - top, 0),
  };
}

export function isSameBox(a: Box | null, b: Box | null) {
  if (a === b) return true;
  if (!a || !b) return false;
  return (
    Math.abs(a.top - b.top) < 0.5 &&
    Math.abs(a.left - b.left) < 0.5 &&
    Math.abs(a.width - b.width) < 0.5 &&
    Math.abs(a.height - b.height) < 0.5
  );
}
