import { PostV1CompleteOnboardingStepStep } from "@/app/api/__generated__/models/postV1CompleteOnboardingStepStep";

export type TabIntroTab = "agents" | "marketplace" | "build";

// Every call to action a tab intro can offer. A union rather than a bare
// string so a typo lands as a type error instead of a silent extra branch in
// the funnel.
export type TabIntroCta =
  | "see_agents"
  | "browse_featured"
  | "ask_autopilot"
  | "builder_tutorial";

// One onboarding step per tab. The backend record is the source of truth —
// it is what stops the card reappearing on a new browser or device.
export const TAB_INTRO_STEPS: Record<
  TabIntroTab,
  PostV1CompleteOnboardingStepStep
> = {
  agents: PostV1CompleteOnboardingStepStep.AGENTS_TAB_INTRO,
  marketplace: PostV1CompleteOnboardingStepStep.MARKETPLACE_TAB_INTRO,
  build: PostV1CompleteOnboardingStepStep.BUILD_TAB_INTRO,
};

// Local cache of the step above, so the card disappears the instant it is
// dismissed rather than waiting on the round trip — and stays gone if that
// round trip failed. The stored value is the user id, not a boolean: another
// account signing in on the same browser must still get its own intro.
const SEEN_KEY_PREFIX = "autogpt:tab-intro-seen:";

export function peekTabIntroSeen(
  tab: TabIntroTab,
  userId: string | null | undefined,
) {
  if (typeof window === "undefined" || !userId) return false;
  try {
    return window.localStorage.getItem(SEEN_KEY_PREFIX + tab) === userId;
  } catch {
    return false;
  }
}

export function setTabIntroSeen(
  tab: TabIntroTab,
  userId: string | null | undefined,
) {
  if (typeof window === "undefined" || !userId) return;
  try {
    window.localStorage.setItem(SEEN_KEY_PREFIX + tab, userId);
  } catch {
    // Storage can be unavailable (private mode, quota). The server step
    // still records the dismissal, so this is only a latency shortcut.
  }
}

// Everything inside the card that can hold focus, in tab order. Used to keep
// Tab inside an `aria-modal` dialog rather than letting it reach the page
// behind the overlay.
const FOCUSABLE_SELECTOR = [
  "a[href]",
  "button:not([disabled])",
  "input:not([disabled])",
  "select:not([disabled])",
  "textarea:not([disabled])",
  '[tabindex]:not([tabindex="-1"])',
].join(",");

export function getFocusableElements(root: HTMLElement) {
  return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));
}
