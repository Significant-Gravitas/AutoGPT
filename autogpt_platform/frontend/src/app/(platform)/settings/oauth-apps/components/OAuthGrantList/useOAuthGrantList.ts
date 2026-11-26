"use client";

import { useState } from "react";

import { useGetOauthListMyOauthGrants } from "@/app/api/__generated__/endpoints/oauth/oauth";
import type { OAuthGrantInfo } from "@/app/api/__generated__/models/oAuthGrantInfo";

export function useOAuthGrantList() {
  const query = useGetOauthListMyOauthGrants({
    query: {
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });
  const [revokeTarget, setRevokeTarget] = useState<OAuthGrantInfo | null>(null);

  const grants = query.data ?? [];

  return {
    grants,
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
    isEmpty: !query.isLoading && !query.isError && grants.length === 0,
    revokeTarget,
    requestRevoke: setRevokeTarget,
    closeRevokeDialog: () => setRevokeTarget(null),
  };
}
