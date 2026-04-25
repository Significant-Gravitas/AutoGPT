"use client";

import {
  getGetV1ListUserApiKeysQueryKey,
  useGetV1ListUserApiKeys,
} from "@/app/api/__generated__/endpoints/api-keys/api-keys";
import { APIKeyStatus } from "@/app/api/__generated__/models/aPIKeyStatus";

export const API_KEYS_QUERY_KEY = getGetV1ListUserApiKeysQueryKey();

export function useAPIKeysList() {
  const query = useGetV1ListUserApiKeys({
    query: {
      select: (response) =>
        response.status === 200
          ? response.data.filter((key) => key.status === APIKeyStatus.ACTIVE)
          : [],
    },
  });

  const keys = query.data ?? [];

  return {
    keys,
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
    isEmpty: !query.isLoading && !query.isError && keys.length === 0,
  };
}
