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
  const { enabled, ready } = useFlagStatus(Flag.ONBOARDING_TAB_INTROS);
  const step = TAB_INTRO_STEPS[tab];

  const [isFinished, setIsFinished] = useState(() =>
    peekTabIntroSeen(tab, userId),
  );

  useEffect(() => {
    // The user record can arrive after mount — re-check once it does.
    if (peekTabIntroSeen(tab, userId)) setIsFinished(true);
  }, [tab, userId]);

  // `state` is null until the provider has fetched the onboarding record.
  // Waiting for it is what keeps the card from flashing in front of a user
  // who already dismissed it on another device.
  const isOpen =
    canShow &&
    ready &&
    Boolean(enabled) &&
    !isFinished &&
    state !== null &&
    !state.completedSteps.includes(step);

  useEffect(() => {
    if (isOpen) trackTabIntro("tab_intro_shown", { tab });
  }, [isOpen, tab]);

  function finish(cta?: string) {
    if (cta) {
      trackTabIntro("tab_intro_cta_clicked", { tab, cta });
    } else {
      trackTabIntro("tab_intro_dismissed", { tab });
    }
    setTabIntroSeen(tab, userId);
    completeStep(step);
    setIsFinished(true);
  }

  return {
    isOpen,
    // "Got it", Escape, or a click on the backdrop.
    dismiss: () => finish(),
    // Any of the card's calls to action, named so the funnel can tell a
    // Build "ask AutoPilot" apart from a Build "learn it yourself".
    takeAction: (cta: string) => finish(cta),
  };
}
