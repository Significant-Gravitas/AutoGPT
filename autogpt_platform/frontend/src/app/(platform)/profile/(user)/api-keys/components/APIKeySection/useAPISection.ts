"use client";
import {
  getGetV1ListUserApiKeysQueryKey,
  useDeleteV1RevokeApiKey,
  useGetV1ListUserApiKeys,
} from "@/app/api/__generated__/endpoints/api-keys/api-keys";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { getQueryClient } from "@/lib/react-query/queryClient";
import {
  getTeamRequestInit,
  getTeamScopedQueryKey,
} from "@/components/contextual/TeamPicker/helpers";
import { useOrgTeamStore } from "@/services/org-team/store";

export const useAPISection = () => {
  const queryClient = getQueryClient();
  const { toast } = useToast();
  const activeOrgID = useOrgTeamStore((s) => s.activeOrgID);
  const activeTeamID = useOrgTeamStore((s) => s.activeTeamID);
  const apiKeysQueryKey = getTeamScopedQueryKey(
    getGetV1ListUserApiKeysQueryKey(),
    activeOrgID,
    activeTeamID,
  );

  const { data: apiKeys, isLoading } = useGetV1ListUserApiKeys({
    request: getTeamRequestInit(activeTeamID),
    query: {
      queryKey: apiKeysQueryKey,
      select: (res) => okData(res)?.filter((key) => key.status === "ACTIVE"),
    },
  });

  const { mutateAsync: revokeAPIKey, isPending: isDeleting } =
    useDeleteV1RevokeApiKey({
      request: getTeamRequestInit(activeTeamID),
      mutation: {
        onSettled: () => {
          return queryClient.invalidateQueries({
            queryKey: apiKeysQueryKey,
          });
        },
      },
    });

  const handleRevokeKey = async (keyId: string) => {
    try {
      await revokeAPIKey({
        keyId: keyId,
      });

      toast({
        title: "Success",
        description: "AutoGPT Platform API key revoked successfully",
      });
    } catch {
      toast({
        title: "Error",
        description: "Failed to revoke AutoGPT Platform API key",
        variant: "destructive",
      });
    }
  };

  return {
    apiKeys,
    isLoading,
    isDeleting,
    handleRevokeKey,
  };
};
