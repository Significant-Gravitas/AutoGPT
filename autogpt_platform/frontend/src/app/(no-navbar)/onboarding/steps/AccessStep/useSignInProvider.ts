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

async function listSignInProviders(): Promise<string[]> {
  const response = await fetch("/api/auth/sign-in-providers");
  if (!response.ok) throw new Error("Could not list sign-in providers");
  const { providers } = (await response.json()) as { providers: string[] };
  return providers;
}
