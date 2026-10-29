import { TOUR_TARGET_ATTRIBUTE, type TourTarget } from "@/lib/tour-targets";
import type { TourStep } from "./steps";

// Steps that mean "already had the first-run intro"; CAPABILITY_CARDS is the
// old deck, so existing users are not walked through by themselves.
export const TOUR_DONE_STEPS = ["FIRST_RUN_TOUR", "CAPABILITY_CARDS"];

// Local cache of FIRST_RUN_TOUR (keyed by user id), so it stays closed even
// before, or without, the server write.
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

// Missing when not laid out (`display: none`) or off to the side (a closed
// sheet). Above or below the fold is only scrolled away.
export function findTourTarget(target: TourTarget) {
  const nodes = document.querySelectorAll<HTMLElement>(
    `[${TOUR_TARGET_ATTRIBUTE}="${target}"]`,
  );
  for (const node of Array.from(nodes)) {
    const rect = node.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) continue;
    if (rect.right <= 0 || rect.left >= window.innerWidth) continue;
    return node;
  }
  return null;
}

export function isOnScreen(box: Box | null): box is Box {
  return box !== null && box.width >= 1 && box.height >= 1;
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

// One string for the card and for what a screen reader hears.
export function withMenuHint(body: string, viaMenu: boolean) {
  return viaMenu ? `${body} Open the menu to find it.` : body;
}

const SPOTLIGHT_PADDING = 6;
// A long chat list would light up the whole sidebar.
const CAPPED_TARGETS: Partial<Record<TourTarget, number>> = { chats: 220 };

// The walkthrough locks page scrolling, so it scrolls a target that runs past
// the edge in itself.
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

// The collapsed rail is too narrow for the full padding on both sides.
const RAIL_PADDING = 4;

function isInRail(element: HTMLElement) {
  return element.closest('[data-collapsible="icon"]') !== null;
}

// In the rail a sidebar item spans the rail's full width while its icon
// button sits to the left, so ringing the item leaves a gap on the right.
function findRailButton(element: HTMLElement) {
  const buttons = element.querySelectorAll<HTMLElement>(
    '[data-sidebar="menu-button"]',
  );
  for (const button of Array.from(buttons)) {
    const rect = button.getBoundingClientRect();
    if (rect.width >= 1 && rect.height >= 1) return button;
  }
  return null;
}

export function getSpotlightBox(
  element: HTMLElement,
  target: TourTarget | null,
): Box {
  const cap = (target && CAPPED_TARGETS[target]) ?? Number.POSITIVE_INFINITY;
  const inRail = isInRail(element);
  const padding = inRail ? RAIL_PADDING : SPOTLIGHT_PADDING;
  const rect = (
    (inRail && findRailButton(element)) ||
    element
  ).getBoundingClientRect();
  const top = Math.max(rect.top - padding, padding);
  const left = Math.max(rect.left - padding, padding);
  const right = Math.min(rect.right + padding, window.innerWidth - padding);
  const bottom = Math.min(
    rect.bottom + padding,
    window.innerHeight - padding,
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
