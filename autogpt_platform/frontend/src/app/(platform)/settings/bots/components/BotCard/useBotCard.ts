import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import {
  getListBotPlatformsQueryKey,
  useDeletePlatformLinkingUnlinkADmUserLink,
  useDeletePlatformLinkingUnlinkAPlatformServer,
} from "@/app/api/__generated__/endpoints/platform-linking/platform-linking";
import { useToast } from "@/components/molecules/Toast/use-toast";

export function useBotCard() {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  // Track every in-flight link ID independently. A single string would
  // get clobbered when the first of two concurrent unlinks settles,
  // re-enabling the still-pending button and allowing a duplicate DELETE.
  const [pendingIds, setPendingIds] = useState<ReadonlySet<string>>(
    () => new Set(),
  );

  function markPending(linkId: string) {
    setPendingIds((prev) => {
      const next = new Set(prev);
      next.add(linkId);
      return next;
    });
  }

  function clearPending(linkId: string) {
    setPendingIds((prev) => {
      if (!prev.has(linkId)) return prev;
      const next = new Set(prev);
      next.delete(linkId);
      return next;
    });
  }

  const invalidatePlatforms = () =>
    queryClient.invalidateQueries({ queryKey: getListBotPlatformsQueryKey() });

  function showError(action: string) {
    toast({
      title: `Couldn't ${action}`,
      description: "Please try again in a moment.",
      variant: "destructive",
    });
  }

  const unlinkServer = useDeletePlatformLinkingUnlinkAPlatformServer({
    mutation: {
      onSuccess: () => invalidatePlatforms(),
      onError: () => showError("unlink the server"),
    },
  });

  const unlinkDm = useDeletePlatformLinkingUnlinkADmUserLink({
    mutation: {
      onSuccess: () => invalidatePlatforms(),
      onError: () => showError("unlink the DM"),
    },
  });

  function unlinkServerLink(linkId: string) {
    if (pendingIds.has(linkId)) return;
    markPending(linkId);
    unlinkServer.mutate({ linkId }, { onSettled: () => clearPending(linkId) });
  }

  function unlinkDmLink(linkId: string) {
    if (pendingIds.has(linkId)) return;
    markPending(linkId);
    unlinkDm.mutate({ linkId }, { onSettled: () => clearPending(linkId) });
  }

  function isPending(linkId: string) {
    return pendingIds.has(linkId);
  }

  return {
    isPending,
    isUnlinking: unlinkServer.isPending || unlinkDm.isPending,
    unlinkServerLink,
    unlinkDmLink,
  };
}
