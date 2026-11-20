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
  onFinished: (outcome: { wasShown: boolean }) => void;
}

// Opens once on the first home visit after onboarding, without waiting for
// the onboarding record (slow after a sign-up); shuts again if it then shows
// an intro.
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
      finish({ wasShown: false });
      return;
    }
    startedFor.current = userID;
    open("auto");
  }, [isPending, hasHadIntro, userID, open, close, finish]);

  // Reads the store, not `source`: in the render whose effect above opens the
  // walkthrough, `source` is still null.
  useEffect(() => {
    const isClosed = useFirstRunTourStore.getState().source === null;
    // A close while signed out, or for another account, is not this one's.
    if (startedFor.current === userID && isPending && isClosed)
      finish({ wasShown: true });
  }, [isPending, source, userID, finish]);
}
