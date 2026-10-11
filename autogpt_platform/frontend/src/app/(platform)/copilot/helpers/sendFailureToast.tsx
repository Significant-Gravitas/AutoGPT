import { Button } from "@/components/atoms/Button/Button";
import { toast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { toast as sonnerToast } from "sonner";
import { describeSendFailure } from "../components/ChatInput/helpers";

const BILLING_PATH = "/settings/billing";
const PAYWALL_TOAST_KEY = "copilot-send-paywall";
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
  if (isPaywallError(error)) {
    toast({
      title: "Choose a plan to keep chatting",
      description: `Sending needs an active plan — ${paywallRecovery}.`,
      variant: "destructive",
      duration: PAYWALL_TOAST_DURATION_MS,
      replaceKey: PAYWALL_TOAST_KEY,
      action: (
        <Button
          as="NextLink"
          href={BILLING_PATH}
          size="small"
          variant="secondary"
          onClick={() => sonnerToast.dismiss(PAYWALL_TOAST_KEY)}
        >
          See plans
        </Button>
      ),
    });
    return;
  }
  toast({
    title,
    description: describeSendFailure(error, recovery),
    variant: "destructive",
  });
}
