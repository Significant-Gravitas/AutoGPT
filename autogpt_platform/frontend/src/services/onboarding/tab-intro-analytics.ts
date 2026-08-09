// Tab-intro funnel (PRD section 4: "Tab cards: shown, CTA vs dismiss").
// Every event carries the tab it belongs to, so one funnel answers both
// "which tabs get discovered" and "does the card's CTA earn its place".
//
// Separate from brain-dump-analytics because the tab intros ship on their
// own flag and outlive the dump funnel — a shared event union would tie
// two unrelated rollouts together.

import posthog from "posthog-js";

type TabIntroEvent =
  | "tab_intro_shown"
  // The card's primary CTA was used, as opposed to any of the ways out
  // ("Got it", Escape, the backdrop) that all land on `tab_intro_dismissed`.
  | "tab_intro_cta_clicked"
  | "tab_intro_dismissed";

export function trackTabIntro(
  event: TabIntroEvent,
  properties: { tab: string; cta?: string },
) {
  try {
    posthog.capture(event, properties);
  } catch {
    // A blocked analytics host must never break a first visit.
  }
}
