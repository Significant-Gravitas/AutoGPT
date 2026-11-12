import { ImpersonationState } from "@/lib/impersonation";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useSyncExternalStore } from "react";

// Starting or stopping an impersonation reloads the page, so there is nothing
// to subscribe to.
function subscribe() {
  return () => {};
}

function isImpersonating() {
  return Boolean(ImpersonationState.get());
}

function isImpersonatingOnServer() {
  return false;
}

// The backend refuses account deletion to an admin acting as the user.
export function useCanDeleteAccount() {
  const isEnabled = useGetFlag(Flag.SELF_SERVE_ACCOUNT_DELETION);
  const isActingAsUser = useSyncExternalStore(
    subscribe,
    isImpersonating,
    isImpersonatingOnServer,
  );
  return isEnabled && !isActingAsUser;
}
