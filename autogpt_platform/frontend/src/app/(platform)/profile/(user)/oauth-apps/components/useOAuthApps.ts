"use client";

import { useState } from "react";
import {
  useGetOauthListMyOauthApps,
  usePatchOauthUpdateAppStatus,
  usePostOauthUploadAppLogo,
  getGetOauthListMyOauthAppsQueryKey,
} from "@/app/api/__generated__/endpoints/oauth/oauth";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
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

  const { mutateAsync: uploadLogo } = usePostOauthUploadAppLogo({
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
    try {
      setUploadingAppId(appId);
      const result = await uploadLogo({
        appId,
        data: { file },
      });

      if (result.status === 200) {
        toast({
          title: "Success",
          description: "Logo uploaded successfully",
        });
      } else {
        throw new Error("Failed to upload logo");
      }
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
