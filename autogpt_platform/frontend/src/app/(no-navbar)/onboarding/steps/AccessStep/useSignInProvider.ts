import { useQuery } from "@tanstack/react-query";
import { ACCESS_PROVIDERS } from "./accessProviders";

/** The provider this account signs in with whose data it can also connect. */
export function useSignInProvider(enabled: boolean) {
  const providers = useQuery({
    queryKey: ["auth", "sign-in-providers"],
    queryFn: listSignInProviders,
    enabled,
    staleTime: Infinity,
    retry: 1,
  });
  return {
    provider: providers.data?.find((id) => id in ACCESS_PROVIDERS) ?? null,
    // A failed lookup only drops the optional step; it never blocks onboarding.
    isReady: !enabled || !providers.isPending,
  };
}

// As long as the feature flags the wizard also waits on.
const LOOKUP_TIMEOUT_MS = 5000;

async function listSignInProviders(): Promise<string[]> {
  // A lookup that hangs has to fail like one that errors, or onboarding
  // waits on it.
  const timeout = new AbortController();
  const timer = setTimeout(() => timeout.abort(), LOOKUP_TIMEOUT_MS);
  try {
    const response = await fetch("/api/auth/sign-in-providers", {
      signal: timeout.signal,
    });
    if (!response.ok) throw new Error("Could not list sign-in providers");
    const { providers } = (await response.json()) as { providers: string[] };
    return providers;
  } finally {
    clearTimeout(timer);
  }
}
