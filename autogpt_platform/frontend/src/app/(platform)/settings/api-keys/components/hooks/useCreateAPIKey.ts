"use client";

import { useQueryClient } from "@tanstack/react-query";

import { usePostV1CreateNewApiKey } from "@/app/api/__generated__/endpoints/api-keys/api-keys";
import type { CreateAPIKeyRequest } from "@/app/api/__generated__/models/createAPIKeyRequest";
import type { CreateAPIKeyResponse } from "@/app/api/__generated__/models/createAPIKeyResponse";
import { toast } from "@/components/molecules/Toast/use-toast";

import { API_KEYS_QUERY_KEY } from "./useAPIKeysList";

export function useCreateAPIKey() {
  const queryClient = useQueryClient();

  const mutation = usePostV1CreateNewApiKey({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({ queryKey: API_KEYS_QUERY_KEY });
        toast({ title: "API key created", variant: "success" });
      },
      onError: (error) => {
        toast({
          title: "Failed to create API key",
          description: error instanceof Error ? error.message : undefined,
          variant: "destructive",
        });
      },
    },
  });

  async function createKey(payload: CreateAPIKeyRequest) {
    // The custom Orval mutator throws on non-2xx, so reaching this line
    // guarantees the success variant of the discriminated union.
    const response = await mutation.mutateAsync({ data: payload });
    return response.data as CreateAPIKeyResponse;
  }

  return {
    createKey,
    isPending: mutation.isPending,
  };
}
