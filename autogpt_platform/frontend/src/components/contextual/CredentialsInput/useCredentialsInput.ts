import { useDeleteV1DeleteCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
import useCredentials from "@/hooks/useCredentials";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { postV2InitiateOauthLoginForAnMcpServer } from "@/app/api/__generated__/endpoints/mcp/mcp";
import { openOAuthPopup } from "@/lib/oauth-popup";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import {
  filterSystemCredentials,
  getActionButtonText,
  getSystemCredentials,
} from "./helpers";

export type CredentialsInputState = ReturnType<typeof useCredentialsInput>;

type Params = {
  schema: BlockIOCredentialsSubSchema;
  selectedCredential?: CredentialsMetaInput;
  onSelectCredential: (newValue?: CredentialsMetaInput) => void;
  siblingInputs?: Record<string, any>;
  onLoaded?: (loaded: boolean) => void;
  readOnly?: boolean;
  isOptional?: boolean;
};

export function useCredentialsInput({
  schema,
  selectedCredential,
  onSelectCredential,
  siblingInputs,
  onLoaded,
  readOnly = false,
  isOptional = false,
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
  const hasAttemptedAutoSelect = useRef(false);
  const oauthAbortRef = useRef<((reason?: string) => void) | null>(null);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      oauthAbortRef.current?.();
    };
  }, []);

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

  // Unselect credential if not available in the loaded credential list.
  // Skip when no credentials have been loaded yet (empty list could mean
  // the provider data hasn't finished loading, not that the credential is invalid).
  useEffect(() => {
    if (readOnly) return;
    if (!credentials || !("savedCredentials" in credentials)) return;
    const availableCreds = credentials.savedCredentials;
    if (availableCreds.length === 0) return;
    if (
      selectedCredential &&
      !availableCreds.some((c) => c.id === selectedCredential.id)
    ) {
      onSelectCredential(undefined);
      // Reset auto-selection flag so it can run again after unsetting invalid credential
      hasAttemptedAutoSelect.current = false;
    }
  }, [credentials, selectedCredential, onSelectCredential, readOnly]);

  // Auto-select the first available credential on initial mount
  // Once a user has made a selection, we don't override it
  useEffect(
    function autoSelectCredential() {
      if (readOnly) return;
      if (!credentials || !("savedCredentials" in credentials)) return;
      if (selectedCredential?.id) return;

      const savedCreds = credentials.savedCredentials;
      if (savedCreds.length === 0) return;

      if (hasAttemptedAutoSelect.current) return;
      hasAttemptedAutoSelect.current = true;

      // Auto-select if exactly one credential matches.
      // For optional fields with multiple options, let the user choose.
      if (isOptional && savedCreds.length > 1) return;

      const cred = savedCreds[0];
      onSelectCredential({
        id: cred.id,
        type: cred.type,
        provider: credentials.provider,
        title: (cred as any).title,
      });
    },
    [
      credentials,
      selectedCredential?.id,
      readOnly,
      isOptional,
      onSelectCredential,
    ],
  );

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
    mcpOAuthCallback,
    isSystemProvider,
    discriminatorValue,
  } = credentials;

  // Split credentials into user and system
  const userCredentials = filterSystemCredentials(savedCredentials);
  const systemCredentials = getSystemCredentials(savedCredentials);

  async function handleOAuthLogin() {
    setOAuthError(null);

    // Abort any previous OAuth flow
    oauthAbortRef.current?.();

    // MCP uses dynamic OAuth discovery per server URL
    const isMCP = provider === "mcp" && !!discriminatorValue;

    try {
      let login_url: string;
      let state_token: string;

      if (isMCP) {
        const mcpLoginResponse = await postV2InitiateOauthLoginForAnMcpServer({
          server_url: discriminatorValue!,
        });
        if (mcpLoginResponse.status !== 200) throw mcpLoginResponse.data;
        ({ login_url, state_token } = mcpLoginResponse.data);
      } else {
        ({ login_url, state_token } = await api.oAuthLogin(
          provider,
          schema.credentials_scopes,
        ));
      }

      setOAuth2FlowInProgress(true);

      const { promise, cleanup } = openOAuthPopup(login_url, {
        stateToken: state_token,
        useCrossOriginListeners: isMCP,
        // Standard OAuth uses "oauth_popup_result", MCP uses "mcp_oauth_result"
        acceptMessageTypes: isMCP
          ? ["mcp_oauth_result"]
          : ["oauth_popup_result"],
      });

      oauthAbortRef.current = cleanup.abort;
      // Expose abort signal for the waiting modal's cancel button
      const controller = new AbortController();
      cleanup.signal.addEventListener("abort", () =>
        controller.abort("completed"),
      );
      setOAuthPopupController(controller);

      const result = await promise;

      // Exchange code for tokens via the provider (updates credential cache)
      const credentialResult = isMCP
        ? await mcpOAuthCallback(result.code, state_token)
        : await oAuthCallback(result.code, result.state);

      // Check if the credential's scopes match the required scopes (skip for MCP)
      if (!isMCP) {
        const requiredScopes = schema.credentials_scopes;
        if (requiredScopes && requiredScopes.length > 0) {
          const grantedScopes = new Set(credentialResult.scopes || []);
          const hasAllRequiredScopes = new Set(requiredScopes).isSubsetOf(
            grantedScopes,
          );

          if (!hasAllRequiredScopes) {
            setOAuthError(
              "Connection failed: the granted permissions don't match what's required. " +
                "Please contact the application administrator.",
            );
            return;
          }
        }
      }

      onSelectCredential({
        id: credentialResult.id,
        type: "oauth2",
        title: credentialResult.title,
        provider,
      });
    } catch (error) {
      if (error instanceof Error && error.message === "OAuth flow timed out") {
        setOAuthError("OAuth flow timed out");
      } else {
        setOAuthError(
          `OAuth error: ${
            error instanceof Error ? error.message : String(error)
          }`,
        );
      }
    } finally {
      setOAuth2FlowInProgress(false);
      oauthAbortRef.current = null;
    }
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
    isSystemProvider,
    userCredentials,
    systemCredentials,
    allCredentials: savedCredentials,
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
      userCredentials.length > 0,
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
