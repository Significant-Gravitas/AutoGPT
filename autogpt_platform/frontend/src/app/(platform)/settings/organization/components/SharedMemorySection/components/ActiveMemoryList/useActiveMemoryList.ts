"use client";

import {
  useDeleteV2RevokeAnActiveSharedMemory,
  useGetV2ListActiveSharedMemories,
} from "@/app/api/__generated__/endpoints/memory/memory";
import type { ActiveMemoryListResponse } from "@/app/api/__generated__/models/activeMemoryListResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useState } from "react";

export function useActiveMemoryList(orgId: string) {
  const [selectedMemoryId, setSelectedMemoryId] = useState<string | null>(null);
  const query = useGetV2ListActiveSharedMemories(orgId, undefined, {
    query: {
      enabled: Boolean(orgId),
      select: (response) => response.data as ActiveMemoryListResponse,
    },
  });
  const revoke = useDeleteV2RevokeAnActiveSharedMemory({
    mutation: {
      onError(error) {
        toast({
          title: "Could not revoke memory",
          description:
            error instanceof Error ? error.message : "Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  async function confirmRevoke() {
    if (!selectedMemoryId) return;
    try {
      await revoke.mutateAsync({ orgId, memoryId: selectedMemoryId });
      setSelectedMemoryId(null);
      toast({ title: "Shared memory revoked", variant: "success" });
      await query.refetch();
    } catch {
      return;
    }
  }

  return {
    items: query.data?.items ?? [],
    isLoading: query.isLoading,
    isError: query.isError,
    refetch: query.refetch,
    selectedMemoryId,
    setSelectedMemoryId,
    confirmRevoke,
    isRevoking: revoke.isPending,
  };
}
