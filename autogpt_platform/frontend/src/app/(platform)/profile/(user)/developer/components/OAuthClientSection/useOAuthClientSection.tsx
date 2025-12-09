"use client";
import { useState } from "react";
import {
  getGetOauthClientsListClientsQueryKey,
  useDeleteOauthClientsDeleteClient,
  useGetOauthClientsListClients,
  usePatchOauthClientsUpdateClient,
  usePostOauthClientsActivateClient,
  usePostOauthClientsRotateWebhookSecret,
  usePostOauthClientsSuspendClient,
} from "@/app/api/__generated__/endpoints/oauth-clients/oauth-clients";
import type { ClientResponse } from "@/app/api/__generated__/models/clientResponse";
import type { UpdateClientRequest } from "@/app/api/__generated__/models/updateClientRequest";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { getQueryClient } from "@/lib/react-query/queryClient";

export function useOAuthClientSection() {
  const queryClient = getQueryClient();
  const { toast } = useToast();

  const [webhookSecretDialogOpen, setWebhookSecretDialogOpen] = useState(false);
  const [newWebhookSecret, setNewWebhookSecret] = useState<string | null>(null);
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [editingClient, setEditingClient] = useState<ClientResponse | null>(
    null,
  );
  const [editFormState, setEditFormState] = useState<UpdateClientRequest>({});

  const { data: oauthClients, isLoading } = useGetOauthClientsListClients({
    query: {
      select: (res) => {
        if (res.status !== 200) return undefined;
        return res.data;
      },
    },
  });

  const { mutateAsync: deleteClient, isPending: isDeleting } =
    useDeleteOauthClientsDeleteClient({
      mutation: {
        onSettled: () => {
          return queryClient.invalidateQueries({
            queryKey: getGetOauthClientsListClientsQueryKey(),
          });
        },
      },
    });

  const { mutateAsync: suspendClient, isPending: isSuspending } =
    usePostOauthClientsSuspendClient({
      mutation: {
        onSettled: () => {
          return queryClient.invalidateQueries({
            queryKey: getGetOauthClientsListClientsQueryKey(),
          });
        },
      },
    });

  const { mutateAsync: activateClient, isPending: isActivating } =
    usePostOauthClientsActivateClient({
      mutation: {
        onSettled: () => {
          return queryClient.invalidateQueries({
            queryKey: getGetOauthClientsListClientsQueryKey(),
          });
        },
      },
    });

  const {
    mutateAsync: rotateWebhookSecret,
    isPending: isRotatingWebhookSecret,
  } = usePostOauthClientsRotateWebhookSecret();

  const { mutateAsync: updateClient, isPending: isUpdating } =
    usePatchOauthClientsUpdateClient({
      mutation: {
        onSettled: () => {
          return queryClient.invalidateQueries({
            queryKey: getGetOauthClientsListClientsQueryKey(),
          });
        },
      },
    });

  async function handleDeleteClient(clientId: string) {
    try {
      await deleteClient({ clientId });
      toast({
        title: "Success",
        description: "OAuth client deleted successfully",
      });
    } catch {
      toast({
        title: "Error",
        description: "Failed to delete OAuth client",
        variant: "destructive",
      });
    }
  }

  async function handleSuspendClient(clientId: string) {
    try {
      await suspendClient({ clientId });
      toast({
        title: "Success",
        description: "OAuth client suspended successfully",
      });
    } catch {
      toast({
        title: "Error",
        description: "Failed to suspend OAuth client",
        variant: "destructive",
      });
    }
  }

  async function handleActivateClient(clientId: string) {
    try {
      await activateClient({ clientId });
      toast({
        title: "Success",
        description: "OAuth client activated successfully",
      });
    } catch {
      toast({
        title: "Error",
        description: "Failed to activate OAuth client",
        variant: "destructive",
      });
    }
  }

  async function handleRotateWebhookSecret(clientId: string) {
    try {
      const response = await rotateWebhookSecret({ clientId });
      if (response.status === 200) {
        setNewWebhookSecret(response.data.webhook_secret);
        setWebhookSecretDialogOpen(true);
        toast({
          title: "Success",
          description: "Webhook secret rotated successfully",
        });
      }
    } catch {
      toast({
        title: "Error",
        description: "Failed to rotate webhook secret",
        variant: "destructive",
      });
    }
  }

  function handleCopyWebhookSecret() {
    if (newWebhookSecret) {
      navigator.clipboard.writeText(newWebhookSecret);
      toast({
        title: "Copied",
        description: "Webhook secret copied to clipboard",
      });
    }
  }

  function handleEditClient(client: ClientResponse) {
    setEditingClient(client);
    setEditFormState({
      name: client.name,
      description: client.description ?? undefined,
      homepage_url: client.homepage_url ?? undefined,
      privacy_policy_url: client.privacy_policy_url ?? undefined,
      terms_of_service_url: client.terms_of_service_url ?? undefined,
      redirect_uris: client.redirect_uris,
      webhook_domains: client.webhook_domains,
    });
    setEditDialogOpen(true);
  }

  async function handleSaveClient() {
    if (!editingClient) return;

    try {
      await updateClient({
        clientId: editingClient.client_id,
        data: editFormState,
      });
      toast({
        title: "Success",
        description: "OAuth client updated successfully",
      });
      setEditDialogOpen(false);
      setEditingClient(null);
    } catch {
      toast({
        title: "Error",
        description: "Failed to update OAuth client",
        variant: "destructive",
      });
    }
  }

  return {
    oauthClients,
    isLoading,
    isDeleting,
    isSuspending,
    isActivating,
    isRotatingWebhookSecret,
    isUpdating,
    handleDeleteClient,
    handleSuspendClient,
    handleActivateClient,
    handleRotateWebhookSecret,
    handleCopyWebhookSecret,
    handleEditClient,
    handleSaveClient,
    webhookSecretDialogOpen,
    setWebhookSecretDialogOpen,
    newWebhookSecret,
    editDialogOpen,
    setEditDialogOpen,
    editingClient,
    editFormState,
    setEditFormState,
  };
}
