"use client";

import { useQueryClient } from "@tanstack/react-query";

import {
  getGetOauthListMyOauthGrantsQueryKey,
  useDeleteOauthRevokeMyOauthGrant,
} from "@/app/api/__generated__/endpoints/oauth/oauth";
import type { OAuthGrantInfo } from "@/app/api/__generated__/models/oAuthGrantInfo";
import { toast } from "@/components/molecules/Toast/use-toast";

export function useRevokeOAuthGrant() {
  const queryClient = useQueryClient();
  const { mutateAsync, isPending } = useDeleteOauthRevokeMyOauthGrant();

  async function revoke(grant: OAuthGrantInfo): Promise<boolean> {
    try {
      await mutateAsync({ appId: grant.application_id });
      toast({
        title: "Access revoked",
        description: `${grant.name} can no longer use your account.`,
        variant: "success",
      });
      return true;
    } catch (error) {
      toast({
        title: `Couldn't revoke access for ${grant.name}`,
        description: error instanceof Error ? error.message : undefined,
        variant: "destructive",
      });
      return false;
    } finally {
      await queryClient.invalidateQueries({
        queryKey: getGetOauthListMyOauthGrantsQueryKey(),
      });
    }
  }

  return { revoke, isPending };
}
