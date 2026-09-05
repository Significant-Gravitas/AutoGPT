import { usePostTrialsConfirmTrial } from "@/app/api/__generated__/endpoints/trials/trials";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
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
        if (!response.data.active && !response.data.converted)
          throw new Error(
            "Your trial is not active. Review your card setup and try again.",
          );
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
