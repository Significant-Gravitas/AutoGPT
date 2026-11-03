import { useGetPlatformConfig } from "@/app/api/__generated__/endpoints/platform-config/platform-config";

// The backend's ENABLE_PLATFORM_PAYMENT, so the paywall UI agrees with enforcement.
// A deployment setting, so it is read once per session; two quick retries keep
// readiness within ~3 s, and isError says the answer is unknown, not off.
export function usePlatformPayment() {
  const { data, isFetched, isError } = useGetPlatformConfig({
    query: {
      staleTime: Infinity,
      retry: 2,
      retryDelay: 1000,
      select: (response) =>
        response.status === 200 && response.data.payment_enabled,
    },
  });
  return { isPaymentEnabled: data === true, isResolved: isFetched, isError };
}
