"use client";
import {
  getGetOauthClientsListClientsQueryKey,
  usePostOauthClientsRegisterClient,
  usePostOauthClientsRotateClientSecret,
} from "@/app/api/__generated__/endpoints/oauth-clients/oauth-clients";
import { ClientSecretResponse } from "@/app/api/__generated__/models/clientSecretResponse";
import { RegisterClientRequest } from "@/app/api/__generated__/models/registerClientRequest";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useState } from "react";

type ClientType = "public" | "confidential";

// Extended type to include webhook_secret (will be in generated types after API regeneration)
interface ClientSecretResponseWithWebhook extends ClientSecretResponse {
  webhook_secret?: string;
}

interface ClientFormState {
  name: string;
  description: string;
  redirectUris: string;
  clientType: ClientType;
  homepageUrl: string;
  privacyPolicyUrl: string;
  termsOfServiceUrl: string;
}

const initialFormState: ClientFormState = {
  name: "",
  description: "",
  redirectUris: "",
  clientType: "public",
  homepageUrl: "",
  privacyPolicyUrl: "",
  termsOfServiceUrl: "",
};

export function useOAuthClientModals() {
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [isSecretDialogOpen, setIsSecretDialogOpen] = useState(false);
  const [formState, setFormState] = useState<ClientFormState>(initialFormState);
  const [newClientSecret, setNewClientSecret] =
    useState<ClientSecretResponseWithWebhook | null>(null);

  const queryClient = getQueryClient();
  const { toast } = useToast();

  const { mutateAsync: registerClient, isPending: isCreating } =
    usePostOauthClientsRegisterClient({
      mutation: {
        onSettled: () => {
          return queryClient.invalidateQueries({
            queryKey: getGetOauthClientsListClientsQueryKey(),
          });
        },
      },
    });

  const { mutateAsync: rotateSecret, isPending: isRotating } =
    usePostOauthClientsRotateClientSecret({
      mutation: {
        onSettled: () => {
          return queryClient.invalidateQueries({
            queryKey: getGetOauthClientsListClientsQueryKey(),
          });
        },
      },
    });

  function resetForm() {
    setFormState(initialFormState);
  }

  async function handleCreateClient() {
    // Parse redirect URIs (comma or newline separated)
    const redirectUris = formState.redirectUris
      .split(/[,\n]/)
      .map((uri) => uri.trim())
      .filter((uri) => uri.length > 0);

    if (redirectUris.length === 0) {
      toast({
        title: "Error",
        description: "At least one redirect URI is required",
        variant: "destructive",
      });
      return;
    }

    if (!formState.name.trim()) {
      toast({
        title: "Error",
        description: "Client name is required",
        variant: "destructive",
      });
      return;
    }

    try {
      const requestData: RegisterClientRequest = {
        name: formState.name.trim(),
        redirect_uris: redirectUris,
        client_type: formState.clientType,
      };

      if (formState.description.trim()) {
        requestData.description = formState.description.trim();
      }
      if (formState.homepageUrl.trim()) {
        requestData.homepage_url = formState.homepageUrl.trim();
      }
      if (formState.privacyPolicyUrl.trim()) {
        requestData.privacy_policy_url = formState.privacyPolicyUrl.trim();
      }
      if (formState.termsOfServiceUrl.trim()) {
        requestData.terms_of_service_url = formState.termsOfServiceUrl.trim();
      }

      const response = await registerClient({
        data: requestData,
      });

      if (response.status === 200) {
        const secretData = response.data as ClientSecretResponseWithWebhook;
        setNewClientSecret(secretData);
        setIsCreateOpen(false);
        setIsSecretDialogOpen(true);
        resetForm();
        toast({
          title: "Success",
          description: "OAuth client created successfully",
        });
      }
    } catch {
      toast({
        title: "Error",
        description: "Failed to create OAuth client",
        variant: "destructive",
      });
    }
  }

  async function handleRotateSecret(clientId: string) {
    try {
      const response = await rotateSecret({ clientId });
      if (response.status === 200) {
        const secretData = response.data as ClientSecretResponseWithWebhook;
        setNewClientSecret(secretData);
        setIsSecretDialogOpen(true);
        toast({
          title: "Success",
          description: "Client secret rotated successfully",
        });
      }
    } catch {
      toast({
        title: "Error",
        description: "Failed to rotate client secret",
        variant: "destructive",
      });
    }
  }

  function handleCopyClientId() {
    if (newClientSecret?.client_id) {
      navigator.clipboard.writeText(newClientSecret.client_id);
      toast({
        title: "Copied",
        description: "Client ID copied to clipboard",
      });
    }
  }

  function handleCopyClientSecret() {
    if (newClientSecret?.client_secret) {
      navigator.clipboard.writeText(newClientSecret.client_secret);
      toast({
        title: "Copied",
        description: "Client secret copied to clipboard",
      });
    }
  }

  function handleCopyWebhookSecret() {
    if (newClientSecret?.webhook_secret) {
      navigator.clipboard.writeText(newClientSecret.webhook_secret);
      toast({
        title: "Copied",
        description: "Webhook secret copied to clipboard",
      });
    }
  }

  function handleSecretDialogChange(open: boolean) {
    setIsSecretDialogOpen(open);
    if (!open) {
      setNewClientSecret(null);
    }
  }

  return {
    isCreateOpen,
    setIsCreateOpen,
    isSecretDialogOpen,
    setIsSecretDialogOpen: handleSecretDialogChange,
    formState,
    setFormState,
    newClientSecret,
    isCreating,
    isRotating,
    handleCreateClient,
    handleRotateSecret,
    handleCopyClientId,
    handleCopyClientSecret,
    handleCopyWebhookSecret,
    resetForm,
  };
}
