"use client";

import { useSessionList } from "@/app/(platform)/copilot/useSessionList";
import { useIsBillingEnabled } from "@/hooks/useIsBillingEnabled";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useOnboarding } from "@/providers/onboarding/onboarding-provider";
import { trackFirstRun } from "@/services/onboarding/first-run-analytics";
import { useEffect, useRef } from "react";
import { useFirstRunTourStore } from "../FirstRunTour/store";
import { useCompetingDialogOpen } from "../WorkflowsMovedNotice/useCompetingDialogOpen";
import { isFirstSecondChat, peekTipSeen, setTipSeen } from "./helpers";
import { requestTip, useContextualTipStore } from "./store";
import { TIPS, type TipID } from "./tips";

// Which one-time tip, if any, is on screen.
export function useContextualTips() {
  const { user, isLoggedIn } = useAuth();
  const userID = user?.id ?? null;
  const { state, completeStep } = useOnboarding();
  const requested = useContextualTipStore((store) => store.requested);
  const remove = useContextualTipStore((store) => store.remove);
  const postpone = useContextualTipStore((store) => store.postpone);
  const shown = useContextualTipStore((store) => store.shown);
  const markShown = useContextualTipStore((store) => store.markShown);
  const putAway = useContextualTipStore((store) => store.putAway);
  const isBillingEnabled = useIsBillingEnabled();
  const isTourOpen = useFirstRunTourStore((store) => store.source !== null);
  const isConnectToolsOpen = useFirstRunTourStore(
    (store) => store.isConnectToolsOpen,
  );

  useSecondChatMoment(isLoggedIn, userID);

  // A tip shown this visit may come back (after a dialog over it closes)
  // until it is put away.
  function isSeen(id: TipID) {
    if (shown.includes(id)) return false;
    return (
      (state?.completedSteps.includes(TIPS[id].step) ?? false) ||
      peekTipSeen(id, userID)
    );
  }

  useEffect(() => {
    if (!state || !userID) return;
    // From the store, not this render: the tip on screen records itself in
    // the same commit and must not be dropped as already had.
    const shownNow = useContextualTipStore.getState().shown;
    requested
      .filter(
        (id) =>
          (TIPS[id].requiresBilling && !isBillingEnabled) ||
          (!shownNow.includes(id) &&
            (state.completedSteps.includes(TIPS[id].step) ||
              peekTipSeen(id, userID))),
      )
      .forEach(remove);
  }, [requested, shown, state, userID, isBillingEnabled, remove]);

  const candidate =
    state && userID
      ? (requested.find(
          (id) =>
            !isSeen(id) && (!TIPS[id].requiresBilling || isBillingEnabled),
        ) ?? null)
      : null;
  const isBlocked = isTourOpen || isConnectToolsOpen;
  const isDialogOpen = useCompetingDialogOpen(candidate !== null && !isBlocked);
  const active = !isBlocked && !isDialogOpen ? candidate : null;

  // Recorded on first showing, so a reload without "Got it" does not bring it
  // back. The card remounts whenever something opens over it; this counts once.
  function reportShown(id: TipID) {
    if (shown.includes(id)) return;
    markShown(id);
    trackFirstRun("contextual_tip_shown", { tip: id });
    setTipSeen(id, userID);
    completeStep(TIPS[id].step);
  }

  function dismiss() {
    if (!active) return;
    trackFirstRun("contextual_tip_dismissed", { tip: active });
    putAway(active);
  }

  return {
    active,
    dismiss,
    reportShown,
    othersWaiting: requested.length > 1,
    postponeActive: () => active && postpone(active),
  };
}

// Watched on the chat list, so every way of starting a chat counts. Goes by
// the most chats seen per account, so a dip back to one does not count.
function useSecondChatMoment(isLoggedIn: boolean, userID: string | null) {
  const { sessions, isLoading } = useSessionList({ enabled: isLoggedIn });
  const mostSeen = useRef<{ userID: string | null; count: number } | null>(
    null,
  );
  const count = sessions.length;

  useEffect(() => {
    if (!isLoggedIn || isLoading) return;
    const seen = mostSeen.current?.userID === userID ? mostSeen.current : null;
    if (isFirstSecondChat(seen?.count ?? null, count))
      requestTip("second-chat");
    mostSeen.current = { userID, count: Math.max(seen?.count ?? 0, count) };
  }, [isLoggedIn, isLoading, count, userID]);
}
