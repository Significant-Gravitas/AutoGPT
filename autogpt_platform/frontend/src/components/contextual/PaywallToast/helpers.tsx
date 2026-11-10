import { toast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import {
  PAYWALL_TOAST_KEY,
  PaywallToastAction,
  PlanRequiredDescription,
} from "./PaywallToast";

export const PAYWALL_TOAST_DURATION_MS = 15_000;

/** The backend answers 402 when the account has no plan (or its trial
 *  lapsed). Retrying can never succeed, so it gets an upgrade path instead of
 *  a "try again". */
export function isPaywallError(error: unknown) {
  return error instanceof ApiError && error.status === 402;
}

/** Shows the plan toast and returns true when `error` is a 402, so a caller
 *  can skip its own failure toast. */
export function toastIfPlanRequired(error: unknown, title: string) {
  if (!isPaywallError(error)) return false;
  toast({
    title,
    description: <PlanRequiredDescription />,
    variant: "destructive",
    duration: PAYWALL_TOAST_DURATION_MS,
    replaceKey: PAYWALL_TOAST_KEY,
    action: <PaywallToastAction />,
  });
  return true;
}
