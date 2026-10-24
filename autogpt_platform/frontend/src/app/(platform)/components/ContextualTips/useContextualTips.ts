"use client";

import { useSessionList } from "@/app/(platform)/copilot/useSessionList";
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

  // A moment for a tip this user already has is simply dropped.
  useEffect(() => {
    if (!state || !userID) return;
    requested
      .filter(
        (id) =>
          state.completedSteps.includes(TIPS[id].step) ||
          peekTipSeen(id, userID),
      )
      .forEach(remove);
  }, [requested, state, userID, remove]);

  const candidate =
    state && userID ? (requested.find((id) => !isSeen(id)) ?? null) : null;
  const isAdmitted = useNoticeAdmission(
    candidate !== null && !isTourOpen && !isConnectToolsOpen,
    userID,
  );
  const active = isAdmitted && !isTourOpen ? candidate : null;

  useEffect(() => {
    if (active) trackFirstRun("contextual_tip_shown", { tip: active });
  }, [active]);

  function dismiss() {
    if (!active) return;
    trackFirstRun("contextual_tip_dismissed", { tip: active });
    setTipSeen(active, userID);
    completeStep(TIPS[active].step);
    remove(active);
  }

  return { active, dismiss };
}

// The chat list going from one chat to two is the first second thread. It is
// watched here rather than where chats are created, so every way of starting
// one counts: the composer, a suggested task, an expert's own chat.
function useSecondChatMoment(isLoggedIn: boolean) {
  const { sessions, isLoading } = useSessionList({ enabled: isLoggedIn });
  const previousCount = useRef<number | null>(null);
  const count = sessions.length;

  useEffect(() => {
    if (!isLoggedIn || isLoading) return;
    if (isFirstSecondChat(previousCount.current, count))
      requestTip("second-chat");
    previousCount.current = count;
  }, [isLoggedIn, isLoading, count]);
}
