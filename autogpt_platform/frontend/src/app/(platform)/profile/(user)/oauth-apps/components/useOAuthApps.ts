"use client";

import { useState, useCallback } from "react";
import {
  useGetOauthListMyOauthApps,
  usePatchOauthUpdateAppStatus,
  usePostOauthUploadAppLogo,
  getGetOauthListMyOauthAppsQueryKey,
} from "@/app/api/__generated__/endpoints/oauth/oauth";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { getQueryClient } from "@/lib/react-query/queryClient";
import BackendAPI from "@/lib/autogpt-server-api";
import type { OAuthApplicationCreationResult } from "@/lib/autogpt-server-api/types";

export const useOAuthApps = () => {
  const queryClient = getQueryClient();
  const { toast } = useToast();
  const [updatingAppId, setUpdatingAppId] = useState<string | null>(null);
  const [uploadingAppId, setUploadingAppId] = useState<string | null>(null);
  const [deletingAppId, setDeletingAppId] = useState<string | null>(null);
  const [regeneratingAppId, setRegeneratingAppId] = useState<string | null>(
    null,
  );
  const [isCreating, setIsCreating] = useState(false);

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

  const invalidateApps = useCallback(() => {
    return queryClient.invalidateQueries({
      queryKey: getGetOauthListMyOauthAppsQueryKey(),
    });
  }, [queryClient]);

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

  const handleCreateApp = async (request: {
    name: string;
    description?: string;
    redirect_uris: string[];
    scopes: string[];
  }): Promise<OAuthApplicationCreationResult | null> => {
    try {
      setIsCreating(true);
      const api = new BackendAPI();
      const result = await api.createMyOAuthApp(request);
      await invalidateApps();
      toast({
        title: "Success",
        description: "OAuth application created successfully",
      });
      return result;
    } catch (error) {
      console.error("Failed to create OAuth app:", error);
      const errorMessage =
        error instanceof Error
          ? error.message
          : "Failed to create OAuth application";
      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
      return null;
    } finally {
      setIsCreating(false);
    }
  };

  const handleDeleteApp = async (appId: string): Promise<boolean> => {
    try {
      setDeletingAppId(appId);
      const api = new BackendAPI();
      await api.deleteMyOAuthApp(appId);
      await invalidateApps();
      toast({
        title: "Success",
        description: "OAuth application deleted successfully",
      });
      return true;
    } catch (error) {
      console.error("Failed to delete OAuth app:", error);
      const errorMessage =
        error instanceof Error
          ? error.message
          : "Failed to delete OAuth application";
      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
      return false;
    } finally {
      setDeletingAppId(null);
    }
  };

  const handleRegenerateSecret = async (
    appId: string,
  ): Promise<string | null> => {
    try {
      setRegeneratingAppId(appId);
      const api = new BackendAPI();
      const result = await api.regenerateMyOAuthSecret(appId);
      toast({
        title: "Success",
        description: "Client secret regenerated successfully",
      });
      return result.client_secret;
    } catch (error) {
      console.error("Failed to regenerate secret:", error);
      const errorMessage =
        error instanceof Error
          ? error.message
          : "Failed to regenerate client secret";
      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
      return null;
    } finally {
      setRegeneratingAppId(null);
    }
  };

  return {
    oauthApps: oauthAppsResponse ?? [],
    isLoading,
    updatingAppId,
    uploadingAppId,
    deletingAppId,
    regeneratingAppId,
    isCreating,
    handleToggleStatus,
    handleUploadLogo,
    handleCreateApp,
    handleDeleteApp,
    handleRegenerateSecret,
  };
};
