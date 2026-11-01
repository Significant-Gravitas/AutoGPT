import { environment } from "@/services/environment";
import {
  Flag,
  useFlagStatus,
  useGetFlag,
} from "@/services/feature-flags/use-get-flag";

// Billing, credits and paywall UI. The hosted platform (a cloud build) always
// shows it. A self-hosted install has no Stripe and no credit ledger by
// default, so it shows it only once its operator turns payments on with
// NEXT_PUBLIC_FORCE_FLAG_ENABLE_PLATFORM_PAYMENT.
export function useIsBillingEnabled() {
  const isPaymentEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);
  return environment.isCloud() || isPaymentEnabled === true;
}

// For gating a whole route: ``isResolved`` stays false until the flag vendor
// has answered or timed out, so a self-hosted install that turns payments on
// through a vendor is not sent to a 404 before the answer arrives, and one
// whose vendor never answers still gets a page.
export function useBillingStatus() {
  const { enabled, ready } = useFlagStatus(Flag.ENABLE_PLATFORM_PAYMENT);
  if (environment.isCloud())
    return { isBillingEnabled: true, isResolved: true };
  return { isBillingEnabled: enabled === true, isResolved: ready };
}
