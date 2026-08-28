"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { deleteV1RevokeApiKey } from "@/app/api/__generated__/endpoints/api-keys/api-keys";
import { toast } from "@/components/molecules/Toast/use-toast";

import { API_KEYS_QUERY_KEY } from "./useAPIKeysList";
import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import type { APIKeyInfo } from "@/app/api/__generated__/models/aPIKeyInfo";

export function useRevokeAPIKey() {
  const queryClient = useQueryClient();
  const [isPending, setIsPending] = useState(false);

  async function revoke(keys: APIKeyInfo[]): Promise<boolean> {
    if (keys.length === 0) return true;

    setIsPending(true);
    try {
      const results = await Promise.allSettled(
        keys.map((key) =>
          deleteV1RevokeApiKey(
            key.id,
            getTenantRequestInit(key.organization_id, key.team_id_restriction),
          ),
        ),
      );
      const failures = results.filter((r) => r.status === "rejected");

      if (failures.length === 0) {
        toast({
          title:
            keys.length === 1
              ? "API key revoked"
              : `${keys.length} API keys revoked`,
          variant: "success",
        });
      } else {
        toast({
          title: "Some API keys could not be revoked",
          description: `${failures.length} of ${keys.length} failed.`,
          variant: "destructive",
        });
      }

      await Promise.all(
        keys.map((key) =>
          queryClient.invalidateQueries({
            queryKey: getTeamScopedQueryKey(
              API_KEYS_QUERY_KEY,
              key.organization_id,
              key.team_id_restriction,
            ),
          }),
        ),
      );

      return failures.length === 0;
    } finally {
      setIsPending(false);
    }
  }

  return { revoke, isPending };
}
