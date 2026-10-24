"use client";

import { useStableCallback } from "@/hooks/useStableCallback";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useOnboarding } from "@/providers/onboarding/onboarding-provider";
import { useEffect, useRef } from "react";
import { peekFirstRunTourSeen, TOUR_DONE_STEPS } from "./helpers";
import { useFirstRunTourStore } from "./store";

interface Args {
  /** Fresh out of onboarding, per the home page's handoff flags. */
  isPending: boolean;
  /** The walkthrough this started has ended, or will never start. */
  onFinished: () => void;
}

// Opens the walkthrough by itself once, on the first home visit after
// onboarding. It does not wait for the onboarding record, which can take a
// while to arrive after a sign-up. The first-run flag is this tab's, so it
// can still outlive its account (signing out and in as someone else): when
// the record shows the walkthrough (FIRST_RUN_TOUR, or the old
// CAPABILITY_CARDS deck), it stays shut, or shuts again if it was opened
// before the record arrived.
export function useFirstRunTourAutoStart({ isPending, onFinished }: Args) {
  const { user } = useAuth();
  const userID = user?.id ?? null;
  const { state } = useOnboarding();
  const source = useFirstRunTourStore((store) => store.source);
  const open = useFirstRunTourStore((store) => store.open);
  const close = useFirstRunTourStore((store) => store.close);
  const startedFor = useRef<string | null>(null);
  const finish = useStableCallback(onFinished);

  const hasHadIntro =
    state !== null &&
    TOUR_DONE_STEPS.some((step) => state.completedSteps.includes(step));

  useEffect(() => {
    if (!isPending || userID === null) return;
    if (startedFor.current === userID) {
      const isAutoOpen = useFirstRunTourStore.getState().source === "auto";
      if (hasHadIntro && isAutoOpen) close();
      return;
    }
    if (hasHadIntro || peekFirstRunTourSeen(userID)) {
      finish();
      return;
    }
    startedFor.current = userID;
    open("auto");
  }, [isPending, hasHadIntro, userID, open, close, finish]);

  // Re-run on every change of `source`, but read the store itself: in the
  // commit where the effect above opened the walkthrough, this render's
  // `source` is still the null from before it opened.
  useEffect(() => {
    const isClosed = useFirstRunTourStore.getState().source === null;
    if (startedFor.current && isPending && isClosed) finish();
  }, [isPending, source, finish]);
}
