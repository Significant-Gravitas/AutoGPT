"use client";

import {
  getGetV1GetUserCreditsQueryKey,
  useGetV1GetUserCredits,
} from "@/app/api/__generated__/endpoints/credits/credits";
import {
  getOrgWalletQueryOptions,
  isForbiddenError,
} from "@/services/org-team/org-wallet";
import { useActiveOrgWallet } from "@/services/org-team/useActiveOrgWallet";

// The tab and its balance card read this one query (one request): in a team
// organization it is the organization's wallet, and a 403 means the viewer
// isn't its owner or a billing manager.
export function useAutomationCreditsBalance() {
  const wallet = useActiveOrgWallet();
  const scoped = getOrgWalletQueryOptions(
    getGetV1GetUserCreditsQueryKey(),
    wallet,
  );

  const query = useGetV1GetUserCredits({
    request: scoped.request,
    query: {
      ...scoped.query,
      select: (res) => {
        const raw = res.data as { credits?: number } | undefined;
        return typeof raw?.credits === "number" ? raw.credits : null;
      },
    },
  });

  return {
    isTeamOrg: wallet.isTeamOrg,
    balanceCents: query.data ?? null,
    isLoading: query.isLoading,
    isError: query.isError,
    isForbidden: wallet.isTeamOrg && isForbiddenError(query.error),
    refetch: query.refetch,
  };
}
