import { toast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { describeSendFailure } from "../components/ChatInput/helpers";
import {
  PAYWALL_TOAST_KEY,
  PaywallToastAction,
  PaywallToastDescription,
} from "../components/PaywallToast/PaywallToast";
import { ReportedSendError } from "./reportedSendError";

const PAYWALL_TOAST_DURATION_MS = 15_000;

/** The backend answers 402 when the account has no plan (or its trial
 *  lapsed). Retrying can never succeed, so it gets an upgrade path instead of
 *  a "try again". */
export function isPaywallError(error: unknown) {
  return error instanceof ApiError && error.status === 402;
}

interface Args {
  title: string;
  error: unknown;
  /** Lowercase clause saying where the user's text went. */
  recovery: string;
  /** Replaces `recovery` on the paywall toast when `recovery` tells the user
   *  to retry, which can't succeed until they have a plan. */
  paywallRecovery?: string;
}

/** The single toast for a send that failed. Every send path reports through
 *  here so one failure is one toast. */
export function toastSendFailure({
  title,
  error,
  recovery,
  paywallRecovery = recovery,
}: Args) {
  if (error instanceof ReportedSendError) return;
  if (isPaywallError(error)) {
    toast({
      title: "Sending needs an active plan",
      description: <PaywallToastDescription recovery={paywallRecovery} />,
      variant: "destructive",
      duration: PAYWALL_TOAST_DURATION_MS,
      replaceKey: PAYWALL_TOAST_KEY,
      action: <PaywallToastAction />,
    });
    return;
  }
  toast({
    title,
    description: describeSendFailure(error, recovery),
    variant: "destructive",
  });
}
