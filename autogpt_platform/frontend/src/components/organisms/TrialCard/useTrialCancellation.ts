import {
  usePostTrialsCancelTrial,
  usePostTrialsResumeTrial,
} from "@/app/api/__generated__/endpoints/trials/trials";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { TrialEvent } from "@/services/analytics/posthog-events";
import { updateTrialStatusCache } from "@/services/trials/updateTrialStatusCache";
import type { useTrialStatus } from "@/services/trials/useTrialStatus";
import { useQueryClient } from "@tanstack/react-query";
import { usePostHog } from "@posthog/react";
import { getTrialDaysLeft } from "./helpers";
import { useTrialCanceledDialog } from "./useTrialCanceledDialog";
import type { useTrialFailure } from "./useTrialFailure";

const CANCEL_FAILED = "Unable to cancel your trial.";
const RESUME_FAILED = "Unable to resume your trial.";

interface Args {
  userID: string | undefined;
  query: ReturnType<typeof useTrialStatus>;
  failure: ReturnType<typeof useTrialFailure>;
}

export function useTrialCancellation({ userID, query, failure }: Args) {
  const queryClient = useQueryClient();
  const posthog = usePostHog();
  const dialog = useTrialCanceledDialog({
    userID,
    endsAt: query.data?.ends_at,
  });
  const { mutateAsync: cancel, isPending: isCanceling } =
    usePostTrialsCancelTrial();
  const { mutateAsync: resume, isPending: isResuming } =
    usePostTrialsResumeTrial();

  async function cancelTrial() {
    if (!userID || isCanceling) return;
    failure.clearFailure();
    try {
      const response = await cancel();
      if (useAuthStore.getState().user?.id !== userID) return;
      if (response.status !== 200) throw new Error(CANCEL_FAILED);
      const trial = response.data;
      await updateTrialStatusCache({
        queryClient,
        userID,
        response,
        onApplied: () => {
          if (isCancelPending(trial))
            dialog.openDialog({ ownerID: userID, endsAt: trial.ends_at });
        },
      });
    } catch (error) {
      failure.reportFailure({ userID, error, fallback: CANCEL_FAILED });
      await query.refetch();
    }
  }

  async function resumeTrial() {
    if (!userID || isResuming) return;
    posthog?.capture(TrialEvent.TRIAL_RESUME_CLICKED, {
      days_left: getTrialDaysLeft(query.data?.ends_at),
    });
    failure.clearFailure();
    try {
      const response = await resume();
      if (useAuthStore.getState().user?.id !== userID) return;
      if (response.status !== 200) throw new Error(RESUME_FAILED);
      await updateTrialStatusCache({ queryClient, userID, response });
    } catch (error) {
      failure.reportFailure({ userID, error, fallback: RESUME_FAILED });
      await query.refetch();
    } finally {
      dialog.closeDialog();
    }
  }

  return {
    isCanceling,
    isResuming,
    cancelTrial,
    resumeTrial,
    showCanceledDialog: dialog.isOpen,
    dismissCanceledDialog: dialog.closeDialog,
    subscribeNow: dialog.subscribeNow,
  };
}

function isCancelPending(trial: TrialStatusResponse) {
  return Boolean(trial.active && trial.cancel_at_period_end);
}
