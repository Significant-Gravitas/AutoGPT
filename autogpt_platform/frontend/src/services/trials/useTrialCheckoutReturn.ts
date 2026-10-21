import { usePostTrialsConfirmTrial } from "@/app/api/__generated__/endpoints/trials/trials";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { trackAdsConversion } from "@/services/analytics/google-ads";
import { useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { updateTrialStatusCache } from "./updateTrialStatusCache";

export function useTrialCheckoutReturn() {
  const userID = useAuthStore((state) => state.user?.id);
  const params = useSearchParams();
  const isReturn = useRef(false);
  if (params.get("trial") === "success") isReturn.current = true;
  const requested = useRef<string | null>(null);
  const reportedTrialStart = useRef<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const [result, setResult] = useState<{
    userID: string;
    trial?: TrialStatusResponse;
    error?: string;
  } | null>(null);
  const queryClient = useQueryClient();
  const { mutateAsync: confirm } = usePostTrialsConfirmTrial();

  useEffect(() => {
    if (
      !isReturn.current ||
      !userID ||
      requested.current === `${userID}:${attempt}`
    )
      return;
    requested.current = `${userID}:${attempt}`;
    confirm()
      .then(async (response) => {
        if (useAuthStore.getState().user?.id !== userID) return;
        if (response.status !== 200)
          throw new Error("Could not confirm your trial.");
        if (!(await updateTrialStatusCache({ queryClient, userID, response })))
          return;
        if (
          !response.data.active &&
          !response.data.converted &&
          response.data.status !== "canceled"
        )
          throw new Error(
            "Your trial is not active. Review your card setup and try again.",
          );
        if (response.data.active && reportedTrialStart.current !== userID) {
          reportedTrialStart.current = userID;
          reportTrialStart(userID);
        }
        setResult({ userID, trial: response.data });
      })
      .catch((error: unknown) => {
        if (useAuthStore.getState().user?.id === userID) {
          setResult({
            userID,
            error:
              error instanceof Error
                ? error.message
                : "Could not confirm your trial.",
          });
        }
      });
  }, [userID, attempt, confirm, queryClient]);

  const current = result && result.userID === userID ? result : null;
  function retry() {
    setResult(null);
    setAttempt((value) => value + 1);
  }
  return {
    ready: !isReturn.current || current !== null,
    error: current?.error,
    active: current?.trial?.active || current?.trial?.converted,
    retry,
  };
}

// The return URL carries no Checkout session id, and a user only ever gets one
// trial, so the user id is the dedup key. Google only receives it with
// advertising consent; for everyone else the latch and the dropped query param
// are what stop a re-render or a reload from counting twice. The param goes
// even when the tag couldn't take the hit: a reload won't unblock it, and
// would only confirm the trial again.
function reportTrialStart(userID: string) {
  trackAdsConversion("trial_started", {
    transactionID: userID,
    email: useAuthStore.getState().user?.email,
  });
  dropTrialReturnParam();
}

function dropTrialReturnParam() {
  const url = new URL(window.location.href);
  if (!url.searchParams.has("trial")) return;
  url.searchParams.delete("trial");
  window.history.replaceState(
    null,
    "",
    `${url.pathname}${url.search}${url.hash}`,
  );
}
