"use client";

import { useSessionList } from "@/app/(platform)/copilot/useSessionList";
import { useBillingStatus } from "@/hooks/useIsBillingEnabled";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useOnboarding } from "@/providers/onboarding/onboarding-provider";
import { useOrgTeamStore } from "@/services/org-team/store";
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
  // Unresolved is not "off": a vendor-held payments flag reads off until it
  // answers, and a billing tip waits for that answer rather than being dropped.
  const { isBillingEnabled, isResolved: isBillingResolved } =
    useBillingStatus();
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
          (TIPS[id].requiresBilling &&
            isBillingResolved &&
            !isBillingEnabled) ||
          (!shownNow.includes(id) &&
            (state.completedSteps.includes(TIPS[id].step) ||
              peekTipSeen(id, userID))),
      )
      .forEach(remove);
  }, [
    requested,
    shown,
    state,
    userID,
    isBillingEnabled,
    isBillingResolved,
    remove,
  ]);

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
// the most chats seen in this list, so a dip back to one does not count. The
// list is per account and per organization, and so is the count; after a
// switch, the new list's baseline waits for data fetched for it, since the
// last list stays on screen for a render before the reset refetches it.
function useSecondChatMoment(isLoggedIn: boolean, userID: string | null) {
  const { sessions, isListLoaded, dataUpdatedAt } = useSessionList({
    enabled: isLoggedIn,
  });
  const orgID = useOrgTeamStore((store) => store.activeOrgID);
  const mostSeen = useRef<ListCount | null>(null);
  const switchedAt = useRef<{ list: ListCount; at: number } | null>(null);
  const count = sessions.length;

  useEffect(() => {
    if (!isLoggedIn || !isListLoaded) return;
    const previous = mostSeen.current;
    const list = { userID, orgID, count };
    if (previous && isAnotherList(previous, list)) {
      if (!switchedAt.current || isAnotherList(switchedAt.current.list, list))
        switchedAt.current = { list, at: Date.now() };
      if (dataUpdatedAt !== undefined && dataUpdatedAt <= switchedAt.current.at)
        return;
    }
    const seen = previous && !isAnotherList(previous, list) ? previous : null;
    if (isFirstSecondChat(seen?.count ?? null, count))
      requestTip("second-chat");
    mostSeen.current = { ...list, count: Math.max(seen?.count ?? 0, count) };
  }, [isLoggedIn, isListLoaded, count, userID, orgID, dataUpdatedAt]);
}

interface ListCount {
  userID: string | null;
  orgID: string | null;
  count: number;
}

// Another account, or one organization replaced by another. The page's org
// arriving for the first time is not a switch: until then the list shown is
// the same one.
function isAnotherList(seen: ListCount, now: ListCount) {
  if (seen.userID !== now.userID) return true;
  return seen.orgID !== null && seen.orgID !== now.orgID;
}
