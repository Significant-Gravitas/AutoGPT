import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { TrialEvent } from "@/services/analytics/posthog-events";
import { usePostHog } from "@posthog/react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { getTrialDaysLeft } from "./helpers";

type EndsAt = TrialStatusResponse["ends_at"];

interface Args {
  userID: string | undefined;
  endsAt: EndsAt;
}

interface OpenArgs {
  ownerID: string;
  endsAt: EndsAt;
}

export function useTrialCanceledDialog({ userID, endsAt }: Args) {
  const posthog = usePostHog();
  const router = useRouter();
  // Opened by this tab's own cancel, never by server state, so a reload or
  // another account never shows it.
  const [openFor, setOpenFor] = useState<string | null>(null);

  function openDialog(args: OpenArgs) {
    setOpenFor(args.ownerID);
    posthog?.capture(TrialEvent.TRIAL_CANCEL_POPUP_VIEWED, {
      days_left: getTrialDaysLeft(args.endsAt),
    });
  }

  function subscribeNow() {
    posthog?.capture(TrialEvent.TRIAL_SUBSCRIBE_NOW_CLICKED, {
      days_left: getTrialDaysLeft(endsAt),
    });
    setOpenFor(null);
    router.push("/settings/billing");
  }

  return {
    isOpen: openFor !== null && openFor === userID,
    openDialog,
    closeDialog: () => setOpenFor(null),
    closeDialogFor: (ownerID: string) =>
      setOpenFor((current) => (current === ownerID ? null : current)),
    subscribeNow,
  };
}
