"use client";

import { useQueryClient } from "@tanstack/react-query";

import { usePostV1CreateNewApiKey } from "@/app/api/__generated__/endpoints/api-keys/api-keys";
import type { CreateAPIKeyRequest } from "@/app/api/__generated__/models/createAPIKeyRequest";
import { toast } from "@/components/molecules/Toast/use-toast";

import { API_KEYS_PAGINATED_QUERY_KEY } from "./useAPIKeysList";

export function useCreateAPIKey() {
  const queryClient = useQueryClient();

  const mutation = usePostV1CreateNewApiKey({
    mutation: {
      onSuccess: (response) => {
        if (response.status === 200) {
          queryClient.invalidateQueries({
            queryKey: API_KEYS_PAGINATED_QUERY_KEY,
          });
          toast({ title: "API key created", variant: "success" });
        }
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
    const response = await mutation.mutateAsync({ data: payload });
    if (response.status !== 200) {
      throw new Error("Failed to create API key");
    }
    return response.data;
  }

  return {
    createKey,
    isPending: mutation.isPending,
  };
}
