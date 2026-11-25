import { useQuery } from "@tanstack/react-query";
import { getV1ListCredentialsByProvider } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { ACCESS_PROVIDERS } from "./accessProviders";

// As long as the sign-in provider lookup the wizard also waits on.
const LOOKUP_TIMEOUT_MS = 5000;

/**
 * Whether the account's sign-in already connected everything its access step
 * would offer, as a Google or Microsoft sign-in that connects leaves it. Then
 * the step is left out.
 */
export function useSignInConnection(
  userID: string | null,
  provider: string | null,
) {
  const access = provider ? ACCESS_PROVIDERS[provider] : undefined;
  const saved = useQuery({
    // Per account, like the provider lookup: kept for good, it must never
    // answer for whoever signs in next in this tab.
    queryKey: ["onboarding", "sign-in-connection", userID, provider],
    queryFn: () => listConnections(provider ?? ""),
    enabled: !!access && userID !== null,
    staleTime: Infinity,
    retry: false,
  });
  return {
    // A failed lookup only keeps the step, which offers the connection.
    isReady: !access || userID === null || !saved.isPending,
    isConnected: !!access && !!saved.data && access.isConnected(saved.data),
  };
}

async function listConnections(provider: string) {
  // A lookup that hangs has to fail like one that errors, or onboarding waits.
  const response = await getV1ListCredentialsByProvider(provider, {
    signal: AbortSignal.timeout(LOOKUP_TIMEOUT_MS),
  });
  return response.status === 200 ? response.data : [];
}
