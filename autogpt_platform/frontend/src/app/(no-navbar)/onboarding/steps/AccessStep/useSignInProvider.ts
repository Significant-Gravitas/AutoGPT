import { useQuery } from "@tanstack/react-query";
import { ACCESS_PROVIDERS } from "./accessProviders";

/** The provider this account signs in with whose data it can also connect. */
export function useSignInProvider(enabled: boolean) {
  const providers = useQuery({
    queryKey: ["auth", "sign-in-providers"],
    queryFn: listSignInProviders,
    enabled,
    staleTime: Infinity,
    // listSignInProviders retries within its own deadline.
    retry: false,
  });
  return {
    provider: providers.data?.find((id) => id in ACCESS_PROVIDERS) ?? null,
    // A failed lookup only drops the optional step; it never blocks onboarding.
    isReady: !enabled || !providers.isPending,
  };
}

// As long as the feature flags the wizard also waits on, for the whole lookup.
const LOOKUP_TIMEOUT_MS = 5000;
const RETRY_DELAY_MS = 1000;

async function listSignInProviders(): Promise<string[]> {
  // One deadline for the lookup and its retry together: a lookup that hangs
  // has to fail like one that errors, or onboarding waits on it.
  const deadline = new AbortController();
  const timer = setTimeout(() => deadline.abort(), LOOKUP_TIMEOUT_MS);
  try {
    return await requestSignInProviders(deadline.signal);
  } catch (error) {
    // A lookup that timed out would only hang again.
    if (deadline.signal.aborted) throw error;
    // One more try after a quick failure, in what is left of the time.
    await waitUnlessAborted(RETRY_DELAY_MS, deadline.signal);
    return await requestSignInProviders(deadline.signal);
  } finally {
    clearTimeout(timer);
  }
}

async function requestSignInProviders(signal: AbortSignal): Promise<string[]> {
  const response = await fetch("/api/auth/sign-in-providers", { signal });
  if (!response.ok) throw new Error("Could not list sign-in providers");
  const { providers } = (await response.json()) as { providers: string[] };
  return providers;
}

function waitUnlessAborted(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(signal.reason);
      },
      { once: true },
    );
  });
}
