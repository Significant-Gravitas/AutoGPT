import { useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import type { ActivationResponse } from "@/app/api/__generated__/models/activationResponse";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { refreshActivation } from "./refreshActivation";
import { safeReturnTo } from "./helpers";
import { storeConsent } from "./storage";

export function useActivationReadiness(
  userID: string | undefined,
  setOpen: (open: boolean) => void,
  setError: (error: string) => void,
) {
  const client = useQueryClient();
  const router = useRouter();
  const [isReady, setReady] = useState(false);
  const [isRefreshing, setRefreshing] = useState(false);
  const finishing = useRef(false);
  async function finish(response: ActivationResponse, notify: boolean) {
    if (!userID || finishing.current) return;
    finishing.current = true;
    setRefreshing(true);
    try {
      await refreshActivation(client, userID);
      if (useAuthStore.getState().user?.id !== userID) return;
      if (notify) setReady(true);
      storeConsent(userID, null);
      if (notify) {
        setOpen(true);
        const returnTo = safeReturnTo(
          response.return_to ?? "/settings/billing",
        );
        if (
          returnTo !==
          `${window.location.pathname}${window.location.search}${window.location.hash}`
        )
          router.replace(returnTo);
      }
    } catch {
      finishing.current = false;
      if (useAuthStore.getState().user?.id === userID) {
        setOpen(true);
        setError(
          "Your payment is confirmed. We’re refreshing your plan and usage before you continue.",
        );
      }
    } finally {
      finishing.current = false;
      setRefreshing(false);
    }
  }
  return { isReady, isRefreshing, finish };
}
