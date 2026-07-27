"use client";

import { useState } from "react";
import {
  useGetOauthListMyOauthApps,
  usePatchOauthUpdateAppStatus,
  getGetOauthListMyOauthAppsQueryKey,
} from "@/app/api/__generated__/endpoints/oauth/oauth";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  isFileTooLarge,
  OAUTH_LOGO_MAX_SIZE_MB,
  uploadOAuthAppLogoDirect,
} from "@/lib/direct-upload";
import { getQueryClient } from "@/lib/react-query/queryClient";

export const useOAuthApps = () => {
  const queryClient = getQueryClient();
  const { toast } = useToast();
  const [updatingAppId, setUpdatingAppId] = useState<string | null>(null);
  const [uploadingAppId, setUploadingAppId] = useState<string | null>(null);

  const { data: oauthAppsResponse, isLoading } = useGetOauthListMyOauthApps({
    query: { select: okData },
  });

  const { mutateAsync: updateStatus } = usePatchOauthUpdateAppStatus({
    mutation: {
      onSettled: () => {
        return queryClient.invalidateQueries({
          queryKey: getGetOauthListMyOauthAppsQueryKey(),
        });
      },
    },
  });

  const handleToggleStatus = async (appId: string, currentStatus: boolean) => {
    try {
      setUpdatingAppId(appId);
      const result = await updateStatus({
        appId,
        data: { is_active: !currentStatus },
      });

      if (result.status === 200) {
        toast({
          title: "Success",
          description: `Application ${result.data.is_active ? "enabled" : "disabled"} successfully`,
        });
      } else {
        throw new Error("Failed to update status");
      }
    } catch {
      toast({
        title: "Error",
        description: "Failed to update application status",
        variant: "destructive",
      });
    } finally {
      setUpdatingAppId(null);
    }
  };

  const handleUploadLogo = async (appId: string, file: File) => {
    if (isFileTooLarge({ file, maxSizeMB: OAUTH_LOGO_MAX_SIZE_MB, toast }))
      return;

    try {
      setUploadingAppId(appId);
      await uploadOAuthAppLogoDirect(appId, file);
      toast({
        title: "Success",
        description: "Logo uploaded successfully",
      });
    } catch (error) {
      console.error("Failed to upload logo:", error);
      const errorMessage =
        error instanceof Error ? error.message : "Failed to upload logo";
      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setUploadingAppId(null);
      void queryClient.invalidateQueries({
        queryKey: getGetOauthListMyOauthAppsQueryKey(),
      });
    }
  };

  return {
    oauthApps: oauthAppsResponse ?? [],
    isLoading,
    updatingAppId,
    uploadingAppId,
    handleToggleStatus,
    handleUploadLogo,
  };
};
