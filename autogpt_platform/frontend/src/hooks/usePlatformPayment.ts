import { useGetPlatformConfig } from "@/app/api/__generated__/endpoints/platform-config/platform-config";

// The backend's ENABLE_PLATFORM_PAYMENT, so the paywall UI agrees with enforcement.
// A deployment setting, so it is read once per session; a failure reads as off
// at once rather than holding onboarding through the retry backoff.
export function usePlatformPayment() {
  const { data, isFetched } = useGetPlatformConfig({
    query: {
      staleTime: Infinity,
      retry: false,
      select: (response) =>
        response.status === 200 && response.data.payment_enabled,
    },
  });
  return { isPaymentEnabled: data === true, isResolved: isFetched };
}
