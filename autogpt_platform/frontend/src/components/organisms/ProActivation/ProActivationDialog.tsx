import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import type { useActivationController } from "@/services/pro-activation/useActivationController";
import { formatAmount } from "@/services/pro-activation/helpers";
import { ActivationTerms } from "./ActivationTerms";
import { ActivationStatus } from "./ActivationStatus";

interface Props {
  activation: ReturnType<typeof useActivationController>;
}

export function ProActivationDialog({ activation }: Props) {
  const {
    attempt,
    isOpen,
    setOpen,
    isBusy,
    isReady,
    error,
    needsNewTerms,
    retryConsent,
    confirm,
    check,
    start,
  } = activation;
  const review = attempt?.status === "confirmation_required" && attempt.terms;
  const title = isReady
    ? "You’re ready to keep going."
    : review
      ? "Make room for what’s next."
      : attempt?.status === "failed"
        ? "Your payment wasn’t completed."
        : attempt?.status === "action_required" ||
            attempt?.status === "payment_required"
          ? "One more step to start Pro."
          : "Getting Pro ready.";
  return (
    <Dialog
      variant="compact"
      controlled={{ isOpen, set: setOpen }}
      styling={{ maxWidth: "560px" }}
      title={
        <div className="space-y-2 pr-7">
          <Text variant="small" className="font-medium text-violet-600">
            {review ? "Start Pro" : "Your Pro upgrade"}
          </Text>
          <span className="block text-[26px] font-semibold leading-[1.2] tracking-[-0.6px] text-zinc-900">
            {title}
          </span>
        </div>
      }
    >
      <Dialog.Content>
        <div className="space-y-5">
          {error && (
            <div
              role="alert"
              className="rounded-large border border-amber-200 bg-amber-50 p-4 text-sm text-zinc-700"
            >
              {error}
            </div>
          )}
          {review ? (
            <>
              <ActivationTerms terms={review} />
              <Button
                variant="primary"
                className="w-full"
                disabled={isBusy || needsNewTerms}
                loading={isBusy}
                onClick={confirm}
              >
                {retryConsent
                  ? "Retry same confirmation"
                  : `Pay ${formatAmount(review.amount_due, review.currency)} & start Pro`}
              </Button>
              <Text variant="small" className="text-center text-zinc-500">
                By confirming, you agree to the charge and renewal terms above.
              </Text>
              {error && !needsNewTerms && retryConsent && (
                <Button
                  variant="secondary"
                  className="w-full"
                  onClick={() => check()}
                  disabled={isBusy}
                >
                  Check existing payment
                </Button>
              )}
              {needsNewTerms && (
                <Button
                  variant="secondary"
                  className="w-full"
                  onClick={() => start(attempt.return_to)}
                  disabled={isBusy}
                >
                  Review current terms
                </Button>
              )}
            </>
          ) : attempt ? (
            <ActivationStatus
              attempt={attempt}
              isBusy={isBusy}
              isReady={isReady}
              check={check}
              close={() => setOpen(false)}
            />
          ) : (
            <Button
              variant="secondary"
              className="w-full"
              loading={isBusy}
              onClick={() => start()}
            >
              {isBusy ? "Loading your upgrade details…" : "Try again"}
            </Button>
          )}
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
