"use client";

import { useSessionList } from "@/app/(platform)/copilot/useSessionList";
import { useIsBillingEnabled } from "@/hooks/useIsBillingEnabled";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useOnboarding } from "@/providers/onboarding/onboarding-provider";
import { trackFirstRun } from "@/services/onboarding/first-run-analytics";
import { useEffect, useRef } from "react";
import { useFirstRunTourStore } from "../FirstRunTour/store";
import { useNoticeAdmission } from "../WorkflowsMovedNotice/useNoticeAdmission";
import { isFirstSecondChat, peekTipSeen, setTipSeen } from "./helpers";
import { requestTip, useContextualTipStore } from "./store";
import { TIPS, type TipID } from "./tips";

// Decides which one-time tip, if any, is on screen. One at a time, never
// over the walkthrough or another dialog, and never twice for a user.
export function useContextualTips() {
  const { user, isLoggedIn } = useAuth();
  const userID = user?.id ?? null;
  const { state, completeStep } = useOnboarding();
  const requested = useContextualTipStore((store) => store.requested);
  const remove = useContextualTipStore((store) => store.remove);
  const postpone = useContextualTipStore((store) => store.postpone);
  const isBillingEnabled = useIsBillingEnabled();
  const isTourOpen = useFirstRunTourStore((store) => store.source !== null);
  const isConnectToolsOpen = useFirstRunTourStore(
    (store) => store.isConnectToolsOpen,
  );

  useSecondChatMoment(isLoggedIn);

  function isSeen(id: TipID) {
    return (
      (state?.completedSteps.includes(TIPS[id].step) ?? false) ||
      peekTipSeen(id, userID)
    );
  }

  // A moment for a tip this user already has, or one this install cannot
  // show (the cost tip points at a wallet that only exists with billing on),
  // is simply dropped rather than left at the head of the queue.
  useEffect(() => {
    if (!state || !userID) return;
    requested
      .filter(
        (id) =>
          (TIPS[id].requiresBilling && !isBillingEnabled) ||
          state.completedSteps.includes(TIPS[id].step) ||
          peekTipSeen(id, userID),
      )
      .forEach(remove);
  }, [requested, state, userID, isBillingEnabled, remove]);

  const candidate =
    state && userID
      ? (requested.find(
          (id) =>
            !isSeen(id) && (!TIPS[id].requiresBilling || isBillingEnabled),
        ) ?? null)
      : null;
  // Admission sticks once granted, so a dialog opening over a tip that is
  // already showing has to take it down here too.
  const isBlocked = isTourOpen || isConnectToolsOpen;
  const isAdmitted = useNoticeAdmission(
    candidate !== null && !isBlocked,
    userID,
  );
  const active = isAdmitted && !isBlocked ? candidate : null;

  // The host outlives the tip card, which comes down and back up whenever
  // admission is checked again (a new page, the walkthrough opening), so it
  // is the one that knows a tip was already seen on screen.
  const shownTips = useRef(new Set<string>());
  function reportShown(id: TipID) {
    const key = `${userID}:${id}`;
    if (shownTips.current.has(key)) return;
    shownTips.current.add(key);
    trackFirstRun("contextual_tip_shown", { tip: id });
  }

  function dismiss() {
    if (!active) return;
    trackFirstRun("contextual_tip_dismissed", { tip: active });
    setTipSeen(active, userID);
    completeStep(TIPS[active].step);
    remove(active);
  }

  return {
    active,
    dismiss,
    reportShown,
    // Nothing on screen to point at yet: let the next tip have its turn.
    postponeActive: () => active && postpone(active),
  };
}

// The chat list growing to two for the first time is the first second
// thread. It is watched here rather than where chats are created, so every
// way of starting one counts: the composer, a suggested task, an expert's
// own chat. It goes by the most chats seen, not the last count, so a list
// that dips back to one (a failed delete, deleting down) does not count.
function useSecondChatMoment(isLoggedIn: boolean) {
  const { sessions, isLoading } = useSessionList({ enabled: isLoggedIn });
  const mostSeen = useRef<number | null>(null);
  const count = sessions.length;

  useEffect(() => {
    if (!isLoggedIn || isLoading) return;
    if (isFirstSecondChat(mostSeen.current, count)) requestTip("second-chat");
    mostSeen.current = Math.max(mostSeen.current ?? 0, count);
  }, [isLoggedIn, isLoading, count]);
}
