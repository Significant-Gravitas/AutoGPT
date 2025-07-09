"use client";
import {
  getGetV1ListUserApiKeysQueryKey,
  usePostV1CreateNewApiKey,
} from "@/app/api/__generated__/endpoints/api-keys/api-keys";
import { APIKeyPermission } from "@/app/api/__generated__/models/aPIKeyPermission";
import { CreateAPIKeyResponse } from "@/app/api/__generated__/models/createAPIKeyResponse";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useState } from "react";

export const useAPIkeysModals = () => {
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [isKeyDialogOpen, setIsKeyDialogOpen] = useState(false);
  const [keyState, setKeyState] = useState({
    newKeyName: "",
    newKeyDescription: "",
    newApiKey: "",
    selectedPermissions: [] as APIKeyPermission[],
  });
  const queryClient = getQueryClient();
  const { toast } = useToast();

  const { mutateAsync: createAPIKey, isPending: isCreating } =
    usePostV1CreateNewApiKey({
      mutation: {
        onSettled: () => {
          return queryClient.invalidateQueries({
            queryKey: getGetV1ListUserApiKeysQueryKey(),
          });
        },
      },
    });

  const handleCreateKey = async () => {
    try {
      const response = await createAPIKey({
        data: {
          name: keyState.newKeyName,
          permissions: keyState.selectedPermissions,
          description: keyState.newKeyDescription,
        },
      });
      setKeyState((prev) => ({
        ...prev,
        newApiKey: (response.data as CreateAPIKeyResponse).plain_text_key,
      }));
      setIsCreateOpen(false);
      setIsKeyDialogOpen(true);
    } catch {
      toast({
        title: "Error",
        description: "Failed to create AutoGPT Platform API key",
        variant: "destructive",
      });
    }
  };

  const handleCopyKey = () => {
    navigator.clipboard.writeText(keyState.newApiKey);
    toast({
      title: "Copied",
      description: "AutoGPT Platform API key copied to clipboard",
    });
  };

  return {
    isCreating,
    handleCreateKey,
    handleCopyKey,
    setIsCreateOpen,
    setIsKeyDialogOpen,
    isCreateOpen,
    isKeyDialogOpen,
    keyState,
    setKeyState,
  };
};
