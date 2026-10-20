// Tab-intro funnel (PRD section 4: "Tab cards: shown, CTA vs dismiss").
// Every event carries the tab it belongs to, so one funnel answers both
// "which tabs get discovered" and "does the card's CTA earn its place".
//
// Separate from brain-dump-analytics because the tab intros outlive the dump
// funnel — a shared event union would tie two unrelated questions together.

import type {
  TabIntroCta,
  TabIntroTab,
} from "@/app/(platform)/components/TabIntroCard/helpers";
import {
  TabIntroEvent,
  type EventName,
} from "@/services/analytics/posthog-events";
import posthog from "posthog-js";

type TabIntroEventName = EventName<typeof TabIntroEvent>;

export function trackTabIntro(
  event: TabIntroEventName,
  properties: { tab: TabIntroTab; cta?: TabIntroCta },
) {
  try {
    posthog.capture(event, properties);
  } catch {
    // A blocked analytics host must never break a first visit.
  }
}
