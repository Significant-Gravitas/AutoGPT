"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import { deleteV1RevokeApiKey } from "@/app/api/__generated__/endpoints/api-keys/api-keys";
import { toast } from "@/components/molecules/Toast/use-toast";

import { API_KEYS_QUERY_KEY } from "./useAPIKeysList";

export function useRevokeAPIKey() {
  const queryClient = useQueryClient();
  const [isPending, setIsPending] = useState(false);

  async function revoke(keyIds: string[]): Promise<boolean> {
    if (keyIds.length === 0) return true;

    setIsPending(true);
    try {
      const results = await Promise.allSettled(
        keyIds.map((id) => deleteV1RevokeApiKey(id)),
      );
      const failures = results.filter((r) => r.status === "rejected");

      if (failures.length === 0) {
        toast({
          title:
            keyIds.length === 1
              ? "API key revoked"
              : `${keyIds.length} API keys revoked`,
          variant: "success",
        });
      } else {
        toast({
          title: "Some API keys could not be revoked",
          description: `${failures.length} of ${keyIds.length} failed.`,
          variant: "destructive",
        });
      }

      await queryClient.invalidateQueries({
        queryKey: API_KEYS_QUERY_KEY,
      });

      return failures.length === 0;
    } finally {
      setIsPending(false);
    }
  }

  return { revoke, isPending };
}
