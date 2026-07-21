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
  OAUTH_ERROR_POPUP_BLOCKED,
  OAUTH_ERROR_WINDOW_CLOSED,
  openOAuthPopup,
  preOpenOAuthPopup,
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
  const [oAuthPopupBlocked, setOAuthPopupBlocked] = useState(false);
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
  const oauthFlowIdRef = useRef(0);
  const preOpenedWindowRef = useRef<Window | null>(null);
  const [isDeletingCredential, setIsDeletingCredential] = useState(false);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      // Invalidate any in-flight flow so its continuation bails instead of
      // adopting the window, and close a window pre-opened by a flow that
      // never reached openOAuthPopup (its abort isn't registered yet).
      oauthFlowIdRef.current += 1;
      oauthAbortRef.current?.();
      if (preOpenedWindowRef.current && !preOpenedWindowRef.current.closed) {
        preOpenedWindowRef.current.close();
      }
      preOpenedWindowRef.current = null;
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
    upgradeableCredentials,
    oAuthCallback,
    mcpOAuthCallback,
    isSystemProvider,
    discriminatorValue,
  } = credentials;

  // Split credentials into user and system
  const userCredentials = filterSystemCredentials(savedCredentials);
  const systemCredentials = getSystemCredentials(savedCredentials);
  const userUpgradeableCredentials = filterSystemCredentials(
    upgradeableCredentials,
  );

  async function executeOAuthFlow(credentialID?: string) {
    setOAuthError(null);

    // Abort any previous OAuth flow, and close the window of one that was
    // still initiating (its abort wasn't registered yet, so the abort above
    // can't reach it).
    oauthAbortRef.current?.();
    if (preOpenedWindowRef.current && !preOpenedWindowRef.current.closed) {
      preOpenedWindowRef.current.close();
    }
    preOpenedWindowRef.current = null;

    // Generation marker — lets this flow's continuation detect that it was
    // superseded (new flow, or unmount) while awaiting the login URL.
    // Re-entrancy contract: SUPERSEDE — the newest call wins because it may
    // target a different credential (add account vs scope upgrade).
    // Deliberately different from useOAuthConnect.connect, which BLOCKS
    // re-entry (single flow target). Reconcile the two explicitly if this
    // lifecycle is ever extracted into a shared hook.
    const flowId = ++oauthFlowIdRef.current;

    // Open the sign-in window synchronously, before the first await — iOS
    // Safari discards the tap's user-gesture context at any async break and
    // then blocks every window.open(), including the new-tab fallback.
    const preOpenedWindow = preOpenOAuthPopup();
    preOpenedWindowRef.current = preOpenedWindow;

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
          credentialID,
        ));
      }

      // A newer flow (or an unmount) superseded this one while the login
      // URL was being fetched — the superseding path already closed the
      // pre-opened window; don't adopt it or touch state.
      if (flowId !== oauthFlowIdRef.current) return;

      setOAuth2FlowInProgress(true);
      setOAuthPopupBlocked(false);

      const { promise, cleanup, popupBlocked, fallbackBlocked } =
        openOAuthPopup(login_url, {
          stateToken: state_token,
          preOpenedWindow,
          // Always enable BroadcastChannel + localStorage listeners — they are
          // the only path that works when the popup is blocked and we fall back
          // to a new tab (window.opener can be severed by cross-origin COOP).
          useCrossOriginListeners: true,
          acceptMessageTypes: isMCP
            ? ["mcp_oauth_result"]
            : ["oauth_popup_result"],
        });
      // Ownership transferred — the helper closes the window on abort now.
      preOpenedWindowRef.current = null;

      // The blank popup window was rejected by the browser — the helper has
      // already fallen back to opening the login URL in a new tab, but that
      // tab is easy to miss. Track the state so the waiting modal can
      // change its copy and direct the user to the right place, and emit a
      // toast in case they've already dismissed the modal.
      // Skip when the fallback was blocked too — the promise has already
      // rejected and the error below carries the correct retry message.
      if (popupBlocked && !fallbackBlocked) {
        setOAuthPopupBlocked(true);
        toast({
          title: "Popup blocked",
          description: OAUTH_ERROR_POPUP_BLOCKED,
        });
      }

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
      // Close the dangling about:blank window only while this flow still
      // owns it — i.e. the error occurred before openOAuthPopup adopted the
      // window. After handoff the ref is nulled and the helper closes the
      // window itself on abort, so it's no longer ours to close.
      if (preOpenedWindowRef.current === preOpenedWindow) {
        preOpenedWindowRef.current = null;
        if (preOpenedWindow && !preOpenedWindow.closed) {
          preOpenedWindow.close();
        }
      }
      // A superseded flow must not surface its errors over the newer flow.
      if (flowId !== oauthFlowIdRef.current) return;
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
      // A superseded flow must not tear down the newer flow's state —
      // nulling oauthAbortRef here would orphan the newer popup's abort
      // handler, and resetting the in-progress flag would close its UI.
      if (flowId === oauthFlowIdRef.current) {
        setOAuth2FlowInProgress(false);
        oauthAbortRef.current = null;
      }
    }
  }

  async function handleOAuthLogin() {
    return executeOAuthFlow();
  }

  async function handleScopeUpgrade(credentialID: string) {
    return executeOAuthFlow(credentialID);
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
    oAuthPopupBlocked,
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
    handleScopeUpgrade,
    userUpgradeableCredentials,
    onSelectCredential,
    schema,
    siblingInputs,
  };
}
