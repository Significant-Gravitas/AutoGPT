"use client";

import {
  useGetV2ListHeldTentativeSharedMemories,
  usePostV2ApproveAHeldMemoryRatifyTentativeActive,
  usePostV2RejectAHeldMemorySoftRetract,
} from "@/app/api/__generated__/endpoints/memory/memory";
import type { HeldMemoryListResponse } from "@/app/api/__generated__/models/heldMemoryListResponse";
import { toast } from "@/components/molecules/Toast/use-toast";

export function useHeldMemoryReviewQueue(orgId: string) {
  const query = useGetV2ListHeldTentativeSharedMemories(orgId, undefined, {
    query: {
      enabled: Boolean(orgId),
      select: (res) => res.data as HeldMemoryListResponse,
    },
  });

  function onError(error: unknown) {
    toast({
      title: "Action failed",
      description: error instanceof Error ? error.message : "Please try again.",
      variant: "destructive",
    });
  }

  const approve = usePostV2ApproveAHeldMemoryRatifyTentativeActive({
    mutation: { onError },
  });
  const reject = usePostV2RejectAHeldMemorySoftRetract({
    mutation: { onError },
  });

  async function handleApprove(memoryId: string) {
    try {
      await approve.mutateAsync({ orgId, memoryId });
      toast({ title: "Memory approved", variant: "success" });
      await query.refetch();
    } catch {
      // onError already surfaced the failure as a toast.
    }
  }

  async function handleReject(memoryId: string) {
    try {
      await reject.mutateAsync({ orgId, memoryId });
      toast({ title: "Memory rejected", variant: "success" });
      await query.refetch();
    } catch {
      // onError already surfaced the failure as a toast.
    }
  }

  return {
    items: query.data?.items ?? [],
    isLoading: query.isLoading,
    isError: query.isError,
    refetch: query.refetch,
    handleApprove,
    handleReject,
    // Mutual-disable: any in-flight approve/reject locks the whole queue so a
    // reviewer can't fire overlapping decisions on the same or other rows.
    isMutating: approve.isPending || reject.isPending,
  };
}
