import { usePlatformPayment } from "@/hooks/usePlatformPayment";
import { environment } from "@/services/environment";

// Billing, credits and paywall UI. The hosted platform (a cloud build) always
// shows it. A self-hosted install has no Stripe and no credit ledger by
// default, so it shows it only once its operator turns payments on with the
// backend's ENABLE_PLATFORM_PAYMENT.
export function useIsBillingEnabled() {
  const { isPaymentEnabled } = usePlatformPayment();
  return environment.isCloud() || isPaymentEnabled;
}

// For gating a whole route: ``isResolved`` stays false until the backend has
// answered, so a self-hosted install that turns payments on is not sent to a
// 404 before the answer arrives.
export function useBillingStatus() {
  const { isPaymentEnabled, isResolved } = usePlatformPayment();
  if (environment.isCloud())
    return { isBillingEnabled: true, isResolved: true };
  return { isBillingEnabled: isPaymentEnabled, isResolved };
}
