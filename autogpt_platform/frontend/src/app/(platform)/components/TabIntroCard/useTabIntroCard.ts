"use client";

import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useOnboarding } from "@/providers/onboarding/onboarding-provider";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { trackTabIntro } from "@/services/onboarding/tab-intro-analytics";
import { useEffect, useState } from "react";
import {
  peekTabIntroSeen,
  setTabIntroSeen,
  TAB_INTRO_STEPS,
  type TabIntroCta,
  type TabIntroTab,
} from "./helpers";

// First-visit gate for a tab's intro card. Opens once per user per tab and
// never again: the onboarding step decides across devices, localStorage
// covers the window before that write lands.
//
// `canShow` lets a tab veto this particular visit without burning the intro —
// the step stays unrecorded, so the next qualifying visit still gets it.
export function useTabIntroCard(tab: TabIntroTab, canShow = true) {
  const { user } = useAuth();
  const userId = user?.id ?? null;
  const { state, completeStep } = useOnboarding();
  // Same gate as the rest of the new onboarding: the tab intros are part of
  // that rollout, so they ship and roll back with it rather than on a flag
  // of their own.
  const { enabled, ready } = useFlagStatus(Flag.ONBOARDING_BRAIN_DUMP);
  const step = TAB_INTRO_STEPS[tab];

  // Who closed the card in this mounted session, rather than a boolean: a
  // second account signing in behind the same mounted hook gets its own intro.
  const [dismissedFor, setDismissedFor] = useState<string | null>(null);

  // `state` is null until the provider has fetched the onboarding record, and
  // `userId` is null until auth resolves. Waiting for both is what keeps the
  // card from flashing in front of a user who already dismissed it — on
  // another device, or on this one before we knew who they were.
  //
  // The cache read is derived here rather than latched into state so it can
  // never lag a render behind the user id, and it sits last so the disabled
  // case never touches localStorage at all.
  const isOpen =
    canShow &&
    ready &&
    Boolean(enabled) &&
    userId !== null &&
    dismissedFor !== userId &&
    state !== null &&
    !state.completedSteps.includes(step) &&
    !peekTabIntroSeen(tab, userId);

  useEffect(() => {
    if (isOpen) trackTabIntro("tab_intro_shown", { tab });
  }, [isOpen, tab]);

  function finish(cta?: TabIntroCta) {
    if (cta) {
      trackTabIntro("tab_intro_cta_clicked", { tab, cta });
    } else {
      trackTabIntro("tab_intro_dismissed", { tab });
    }
    setTabIntroSeen(tab, userId);
    completeStep(step);
    setDismissedFor(userId);
  }

  return {
    isOpen,
    // "Got it", Escape, or a click on the backdrop.
    dismiss: () => finish(),
    // Any of the card's calls to action, named so the funnel can tell a
    // Build "ask AutoPilot" apart from a Build "learn it yourself".
    takeAction: (cta: TabIntroCta) => finish(cta),
  };
}
