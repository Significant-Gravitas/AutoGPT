import type { ActivationResponse } from "@/app/api/__generated__/models/activationResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  CheckmarkCircle02Icon,
  Loading03Icon,
  Shield01Icon,
} from "@hugeicons/core-free-icons";
import { invoiceURL } from "@/services/pro-activation/helpers";

interface Props {
  attempt: ActivationResponse | null;
  isReady: boolean;
  isBusy: boolean;
  check: () => Promise<void>;
  close: () => void;
}

export function ActivationStatus({
  attempt,
  isReady,
  isBusy,
  check,
  close,
}: Props) {
  const url = invoiceURL(attempt?.hosted_invoice_url);
  const payment =
    attempt?.status === "payment_required" ||
    attempt?.status === "action_required";
  const failed = attempt?.status === "failed";
  const needsSupport = ["recovery_required", "terms_changed"].includes(
    attempt?.error_code ?? "",
  );
  return (
    <div className="space-y-6">
      <div className="rounded-xlarge border border-zinc-200 bg-zinc-50 p-5">
        <Icon
          icon={
            isReady
              ? CheckmarkCircle02Icon
              : payment
                ? Shield01Icon
                : Loading03Icon
          }
          size={24}
          className={
            !isReady && !payment && !failed
              ? "mb-4 animate-spin text-violet-600"
              : "mb-4 text-violet-600"
          }
        />
        <Text variant="body" className="text-zinc-700">
          {isReady
            ? "Your Pro plan and usage are ready. Your conversations and work are right where you left them."
            : payment
              ? "Complete payment securely with Stripe using your existing invoice. It opens in a new tab, keeping your draft here while we check your activation."
              : failed
                ? "This payment was canceled or could not be completed. Your usage has not been reset."
                : needsSupport
                  ? "We need to verify your existing payment before you can continue. Contact support; please don’t start another purchase."
                  : "We’re confirming your payment and preparing your Pro allowance. You can close this window; we’ll keep checking."}
        </Text>
      </div>
      {url && payment && (
        <Button
          as="NextLink"
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          variant="primary"
          className="w-full"
        >
          {attempt?.status === "action_required"
            ? "Verify payment securely"
            : "Complete payment securely"}
        </Button>
      )}
      {needsSupport && (
        <Button
          as="NextLink"
          href="mailto:contact@agpt.co"
          variant="primary"
          className="w-full"
        >
          Contact support
        </Button>
      )}
      <Button
        onClick={() => (isReady || failed ? close() : check())}
        disabled={isBusy}
        loading={isBusy}
        variant={isReady ? "primary" : "secondary"}
        className="w-full"
      >
        {isReady
          ? "Continue where you left off"
          : failed
            ? "Back to your work"
            : "Check activation status"}
      </Button>
    </div>
  );
}
