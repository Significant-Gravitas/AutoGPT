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
  const [pendingId, setPendingId] = useState<string | null>(null);

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
      onSuccess: () => {
        invalidatePlatforms();
      },
      onError: () => showError("unlink the server"),
      onSettled: () => setPendingId(null),
    },
  });

  const unlinkDm = useDeletePlatformLinkingUnlinkADmUserLink({
    mutation: {
      onSuccess: () => {
        invalidatePlatforms();
      },
      onError: () => showError("unlink the DM"),
      onSettled: () => setPendingId(null),
    },
  });

  function unlinkServerLink(linkId: string) {
    setPendingId(linkId);
    unlinkServer.mutate({ linkId });
  }

  function unlinkDmLink(linkId: string) {
    setPendingId(linkId);
    unlinkDm.mutate({ linkId });
  }

  return {
    pendingId,
    isUnlinking: unlinkServer.isPending || unlinkDm.isPending,
    unlinkServerLink,
    unlinkDmLink,
  };
}
