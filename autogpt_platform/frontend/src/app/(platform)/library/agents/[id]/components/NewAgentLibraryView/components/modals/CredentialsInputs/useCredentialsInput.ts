import { useDeleteV1DeleteCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
import useCredentials from "@/hooks/useCredentials";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import {
  getActionButtonText,
  OAUTH_TIMEOUT_MS,
  OAuthPopupResultMessage,
} from "./helpers";

export type CredentialsInputState = ReturnType<typeof useCredentialsInput>;

type Params = {
  schema: BlockIOCredentialsSubSchema;
  selectedCredential?: CredentialsMetaInput;
  onSelectCredential: (newValue?: CredentialsMetaInput) => void;
  siblingInputs?: Record<string, any>;
  onLoaded?: (loaded: boolean) => void;
  readOnly?: boolean;
};

export function useCredentialsInput({
  schema,
  selectedCredential,
  onSelectCredential,
  siblingInputs,
  onLoaded,
  readOnly = false,
}: Params) {
  const [isAPICredentialsModalOpen, setAPICredentialsModalOpen] =
    useState(false);
  const [
    isUserPasswordCredentialsModalOpen,
    setUserPasswordCredentialsModalOpen,
  ] = useState(false);
  const [isHostScopedCredentialsModalOpen, setHostScopedCredentialsModalOpen] =
    useState(false);
  const [isOAuth2FlowInProgress, setOAuth2FlowInProgress] = useState(false);
  const [oAuthPopupController, setOAuthPopupController] =
    useState<AbortController | null>(null);
  const [oAuthError, setOAuthError] = useState<string | null>(null);
  const [credentialToDelete, setCredentialToDelete] = useState<{
    id: string;
    title: string;
  } | null>(null);

  const api = useBackendAPI();
  const queryClient = useQueryClient();
  const credentials = useCredentials(schema, siblingInputs);

  const deleteCredentialsMutation = useDeleteV1DeleteCredentials({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: ["/api/integrations/credentials"],
        });
        queryClient.invalidateQueries({
          queryKey: [`/api/integrations/${credentials?.provider}/credentials`],
        });
        setCredentialToDelete(null);
        if (selectedCredential?.id === credentialToDelete?.id) {
          onSelectCredential(undefined);
        }
      },
    },
  });

  useEffect(() => {
    if (onLoaded) {
      onLoaded(Boolean(credentials && credentials.isLoading === false));
    }
  }, [credentials, onLoaded]);

  // Unselect credential if not available
  useEffect(() => {
    if (readOnly) return;
    if (!credentials || !("savedCredentials" in credentials)) return;
    if (
      selectedCredential &&
      !credentials.savedCredentials.some((c) => c.id === selectedCredential.id)
    ) {
      onSelectCredential(undefined);
    }
  }, [credentials, selectedCredential, onSelectCredential, readOnly]);

  // The available credential, if there is only one
  const singleCredential = useMemo(() => {
    if (!credentials || !("savedCredentials" in credentials)) {
      return null;
    }

    return credentials.savedCredentials.length === 1
      ? credentials.savedCredentials[0]
      : null;
  }, [credentials]);

  // Auto-select the one available credential
  useEffect(() => {
    if (readOnly) return;
    if (singleCredential && !selectedCredential) {
      onSelectCredential(singleCredential);
    }
  }, [singleCredential, selectedCredential, onSelectCredential, readOnly]);

  if (
    !credentials ||
    credentials.isLoading ||
    !("savedCredentials" in credentials)
  ) {
    return {
      isLoading: true,
    };
  }

  const {
    provider,
    providerName,
    supportsApiKey,
    supportsOAuth2,
    supportsUserPassword,
    supportsHostScoped,
    savedCredentials,
    oAuthCallback,
  } = credentials;

  async function handleOAuthLogin() {
    setOAuthError(null);
    const { login_url, state_token } = await api.oAuthLogin(
      provider,
      schema.credentials_scopes,
    );
    setOAuth2FlowInProgress(true);
    const popup = window.open(login_url, "_blank", "popup=true");

    if (!popup) {
      throw new Error(
        "Failed to open popup window. Please allow popups for this site.",
      );
    }

    const controller = new AbortController();
    setOAuthPopupController(controller);
    controller.signal.onabort = () => {
      console.debug("OAuth flow aborted");
      setOAuth2FlowInProgress(false);
      popup.close();
    };

    const handleMessage = async (e: MessageEvent<OAuthPopupResultMessage>) => {
      console.debug("Message received:", e.data);
      if (
        typeof e.data != "object" ||
        !("message_type" in e.data) ||
        e.data.message_type !== "oauth_popup_result"
      ) {
        console.debug("Ignoring irrelevant message");
        return;
      }

      if (!e.data.success) {
        console.error("OAuth flow failed:", e.data.message);
        setOAuthError(`OAuth flow failed: ${e.data.message}`);
        setOAuth2FlowInProgress(false);
        return;
      }

      if (e.data.state !== state_token) {
        console.error("Invalid state token received");
        setOAuthError("Invalid state token received");
        setOAuth2FlowInProgress(false);
        return;
      }

      try {
        console.debug("Processing OAuth callback");
        const credentials = await oAuthCallback(e.data.code, e.data.state);
        console.debug("OAuth callback processed successfully");

        // Check if the credential's scopes match the required scopes
        const requiredScopes = schema.credentials_scopes;
        if (requiredScopes && requiredScopes.length > 0) {
          const grantedScopes = new Set(credentials.scopes || []);
          const hasAllRequiredScopes = new Set(requiredScopes).isSubsetOf(
            grantedScopes,
          );

          if (!hasAllRequiredScopes) {
            console.error(
              `Newly created OAuth credential for ${providerName} has insufficient scopes. Required:`,
              requiredScopes,
              "Granted:",
              credentials.scopes,
            );
            setOAuthError(
              "Connection failed: the granted permissions don't match what's required. " +
                "Please contact the application administrator.",
            );
            return;
          }
        }

        onSelectCredential({
          id: credentials.id,
          type: "oauth2",
          title: credentials.title,
          provider,
        });
      } catch (error) {
        console.error("Error in OAuth callback:", error);
        setOAuthError(
          `Error in OAuth callback: ${
            error instanceof Error ? error.message : String(error)
          }`,
        );
      } finally {
        console.debug("Finalizing OAuth flow");
        setOAuth2FlowInProgress(false);
        controller.abort("success");
      }
    };

    console.debug("Adding message event listener");
    window.addEventListener("message", handleMessage, {
      signal: controller.signal,
    });

    setTimeout(() => {
      console.debug("OAuth flow timed out");
      controller.abort("timeout");
      setOAuth2FlowInProgress(false);
      setOAuthError("OAuth flow timed out");
    }, OAUTH_TIMEOUT_MS);
  }

  function handleActionButtonClick() {
    if (supportsOAuth2) {
      handleOAuthLogin();
    } else if (supportsApiKey) {
      setAPICredentialsModalOpen(true);
    } else if (supportsUserPassword) {
      setUserPasswordCredentialsModalOpen(true);
    } else if (supportsHostScoped) {
      setHostScopedCredentialsModalOpen(true);
    }
  }

  function handleCredentialSelect(credentialId: string) {
    const selectedCreds = savedCredentials.find((c) => c.id === credentialId);
    if (selectedCreds) {
      onSelectCredential({
        id: selectedCreds.id,
        type: selectedCreds.type,
        provider: provider,
        title: (selectedCreds as any).title,
      });
    }
  }

  function handleDeleteCredential(credential: { id: string; title: string }) {
    setCredentialToDelete(credential);
  }

  function handleDeleteConfirm() {
    if (credentialToDelete && credentials) {
      deleteCredentialsMutation.mutate({
        provider: credentials.provider,
        credId: credentialToDelete.id,
      });
    }
  }

  return {
    isLoading: false as const,
    provider,
    providerName,
    supportsApiKey,
    supportsOAuth2,
    supportsUserPassword,
    supportsHostScoped,
    credentialsToShow: savedCredentials,
    selectedCredential,
    oAuthError,
    isAPICredentialsModalOpen,
    isUserPasswordCredentialsModalOpen,
    isHostScopedCredentialsModalOpen,
    isOAuth2FlowInProgress,
    oAuthPopupController,
    credentialToDelete,
    deleteCredentialsMutation,
    actionButtonText: getActionButtonText(
      supportsOAuth2,
      supportsApiKey,
      supportsUserPassword,
      supportsHostScoped,
      savedCredentials.length > 0,
    ),
    setAPICredentialsModalOpen,
    setUserPasswordCredentialsModalOpen,
    setHostScopedCredentialsModalOpen,
    setCredentialToDelete,
    handleActionButtonClick,
    handleCredentialSelect,
    handleDeleteCredential,
    handleDeleteConfirm,
    handleOAuthLogin,
    onSelectCredential,
    schema,
    siblingInputs,
  };
}
