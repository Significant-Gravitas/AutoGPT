import useCredentials from "@/hooks/useCredentials";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import {
  BlockIOCredentialsSubSchema,
  CredentialsMetaInput,
} from "@/lib/autogpt-server-api/types";
import { toast } from "@/components/molecules/Toast/use-toast";
import { postV2InitiateOauthLoginForAnMcpServer } from "@/app/api/__generated__/endpoints/mcp/mcp";
import {
  OAUTH_ERROR_FLOW_CANCELED,
  OAUTH_ERROR_FLOW_TIMED_OUT,
  OAUTH_ERROR_WINDOW_CLOSED,
  openOAuthPopup,
} from "@/lib/oauth-popup";
import { useEffect, useRef, useState } from "react";
import {
  countSupportedTypes,
  filterSystemCredentials,
  getActionButtonText,
  getSupportedTypes,
  getSystemCredentials,
  processCredentialDeletion,
  resolveActionTarget,
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
  const [isCredentialTypeSelectorOpen, setCredentialTypeSelectorOpen] =
    useState(false);
  const [isOAuth2FlowInProgress, setOAuth2FlowInProgress] = useState(false);
  const [oAuthError, setOAuthError] = useState<string | null>(null);
  const [credentialToDelete, setCredentialToDelete] = useState<{
    id: string;
    title: string;
  } | null>(null);
  const [deleteWarningMessage, setDeleteWarningMessage] = useState<
    string | null
  >(null);

  const api = useBackendAPI();
  const credentials = useCredentials(schema, siblingInputs);
  const hasAttemptedAutoSelect = useRef(false);
  const oauthAbortRef = useRef<((reason?: string) => void) | null>(null);
  const [isDeletingCredential, setIsDeletingCredential] = useState(false);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      oauthAbortRef.current?.();
    };
  }, []);

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

      // Auto-select only when there is exactly one saved credential.
      // With multiple options the user must choose — regardless of optional/required.
      if (savedCreds.length > 1) return;

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
      const message = error instanceof Error ? error.message : String(error);
      if (
        message === OAUTH_ERROR_WINDOW_CLOSED ||
        message === OAUTH_ERROR_FLOW_CANCELED
      ) {
        // User closed the popup or clicked cancel — not an error
      } else if (message === OAUTH_ERROR_FLOW_TIMED_OUT) {
        setOAuthError(OAUTH_ERROR_FLOW_TIMED_OUT);
      } else {
        setOAuthError(`OAuth error: ${message}`);
      }
    } finally {
      setOAuth2FlowInProgress(false);
      oauthAbortRef.current = null;
    }
  }

  const hasMultipleCredentialTypes =
    countSupportedTypes(
      supportsOAuth2,
      supportsApiKey,
      supportsUserPassword,
      supportsHostScoped,
    ) > 1;

  const supportedTypes = getSupportedTypes(
    supportsOAuth2,
    supportsApiKey,
    supportsUserPassword,
    supportsHostScoped,
  );

  function handleActionButtonClick() {
    const target = resolveActionTarget(
      hasMultipleCredentialTypes,
      supportsOAuth2,
      supportsApiKey,
      supportsUserPassword,
      supportsHostScoped,
    );
    switch (target) {
      case "type_selector":
        setCredentialTypeSelectorOpen(true);
        break;
      case "oauth":
        handleOAuthLogin();
        break;
      case "api_key":
        setAPICredentialsModalOpen(true);
        break;
      case "user_password":
        setUserPasswordCredentialsModalOpen(true);
        break;
      case "host_scoped":
        setHostScopedCredentialsModalOpen(true);
        break;
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

  function cancelOAuthFlow() {
    oauthAbortRef.current?.("canceled");
  }

  function handleDeleteCredential(credential: { id: string; title: string }) {
    setDeleteWarningMessage(null);
    setCredentialToDelete(credential);
  }

  async function handleDeleteConfirm(force: boolean = false) {
    if (
      !credentialToDelete ||
      !credentials ||
      !("deleteCredentials" in credentials)
    )
      return;

    setIsDeletingCredential(true);
    try {
      const state = await processCredentialDeletion(
        credentialToDelete,
        selectedCredential?.id,
        credentials.deleteCredentials,
        force,
      );

      if (state.shouldUnselectCurrent) {
        onSelectCredential(undefined);
      }
      setDeleteWarningMessage(state.warningMessage);
      setCredentialToDelete(state.credentialToDelete);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Something went wrong";
      toast({
        title: "Failed to delete credential",
        description: message,
        variant: "destructive",
      });
    } finally {
      setIsDeletingCredential(false);
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
    hasMultipleCredentialTypes,
    supportedTypes,
    isSystemProvider,
    userCredentials,
    systemCredentials,
    allCredentials: savedCredentials,
    selectedCredential,
    oAuthError,
    isAPICredentialsModalOpen,
    isUserPasswordCredentialsModalOpen,
    isHostScopedCredentialsModalOpen,
    isCredentialTypeSelectorOpen,
    isOAuth2FlowInProgress,
    cancelOAuthFlow,
    credentialToDelete,
    deleteWarningMessage,
    isDeletingCredential,
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
    setCredentialTypeSelectorOpen,
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
