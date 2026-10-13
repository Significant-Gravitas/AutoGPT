import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import type { useActivationController } from "@/services/pro-activation/useActivationController";
import { formatAmount } from "@/services/pro-activation/helpers";
import { ActivationTerms } from "./ActivationTerms";
import { ActivationStatus } from "./ActivationStatus";
import { ActivationPaymentMethod } from "./ActivationPaymentMethod";
import { billingCadence } from "@/services/pro-activation/reviewDetails";
import { useQuoteExpiry } from "@/services/pro-activation/useQuoteExpiry";

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
    hasFreshUsage,
    readyPlan,
    error,
    needsNewTerms,
    retryConsent,
    confirm,
    check,
    start,
  } = activation;
  const review = attempt?.status === "confirmation_required" && attempt.terms;
  const expiry = useQuoteExpiry(review ? review.expires_at : undefined);
  const refreshTerms = needsNewTerms || (expiry.expired && !retryConsent);
  const plan = (attempt?.terms?.plan ?? readyPlan) === "MAX" ? "Max" : "Pro";
  const title = isReady
    ? `${plan} is active`
    : review
      ? `Start ${plan} today`
      : attempt?.status === "not_applicable"
        ? "Your plan has changed"
        : attempt?.status === "failed"
          ? "Your payment wasn’t completed."
          : attempt?.status === "action_required" ||
              attempt?.status === "payment_required"
            ? `One more step to start ${plan}.`
            : `Getting ${plan} ready.`;
  return (
    <Dialog
      variant="compact"
      controlled={{ isOpen, set: setOpen }}
      styling={{ maxWidth: "560px" }}
      title={
        <div className="space-y-2 pr-7">
          <Text variant="small" className="font-medium text-zinc-500">
            {review ? "Payment review" : `AutoGPT ${plan}`}
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
              <ActivationPaymentMethod />
              {expiry.expiringSoon && !retryConsent && (
                <Text variant="small" className="text-amber-700">
                  This price is held for a few more minutes.
                </Text>
              )}
              {expiry.expired && !retryConsent && (
                <Text variant="small" className="text-amber-700" role="status">
                  This quote has expired. Review the current terms before
                  paying.
                </Text>
              )}
              <Button
                variant="primary"
                className="w-full"
                disabled={isBusy || refreshTerms}
                loading={isBusy}
                onClick={confirm}
              >
                {retryConsent
                  ? "Retry same confirmation"
                  : `Pay ${formatAmount(review.amount_due, review.currency)} and start ${plan}`}
              </Button>
              <Text variant="small" className="text-center text-zinc-500">
                Renews {billingCadence(review)} until you cancel. Cancel anytime
                in Billing.
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
              {refreshTerms && (
                <Button
                  variant="secondary"
                  className="w-full"
                  onClick={() => start(attempt.return_to)}
                  disabled={isBusy}
                >
                  Review current terms
                </Button>
              )}
              <Button
                variant="link"
                className="w-full text-zinc-600"
                onClick={() => setOpen(false)}
              >
                Not now
              </Button>
            </>
          ) : attempt ? (
            <ActivationStatus
              attempt={attempt}
              isBusy={isBusy}
              isReady={isReady}
              hasFreshUsage={hasFreshUsage}
              plan={plan}
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
