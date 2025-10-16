"use client";

import { Button } from "@/components/atoms/Button/Button";
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import { useState } from "react";
import { startOAuthPopupFlow } from "../../helpers";
import { OAuthFlowWaitingModal } from "@/app/(platform)/library/agents/[id]/components/AgentRunsView/components/OAuthWaitingModal/OAuthWaitingModal";

type Props = {
  provider: string;
  providerName: string;
  scopes?: string[];
  buttonText?: string;
  disabled?: boolean;
  onSuccess?: () => void;
  onError?: (message: string) => void;
  /** Callback invoked after user cancels the waiting modal */
  onCancel?: () => void;
  /** Callback to exchange auth code + state for credentials */
  oAuthCallback: (code: string, state: string) => Promise<unknown>;
};

export function OAuthLogin(props: Props) {
  const {
    provider,
    providerName,
    scopes,
    buttonText,
    disabled,
    onSuccess,
    onError,
    onCancel,
    oAuthCallback,
  } = props;

  const api = useBackendAPI();
  const [isOAuth2FlowInProgress, setOAuth2FlowInProgress] = useState(false);
  const [oAuthError, setOAuthError] = useState<string | null>(null);
  const [oAuthPopupAbort, setOAuthPopupAbort] = useState<null | (() => void)>(
    null,
  );

  async function handleOAuthLogin() {
    setOAuthError(null);
    const { login_url, state_token } = await api.oAuthLogin(provider, scopes);
    setOAuth2FlowInProgress(true);
    const { promise, abort } = startOAuthPopupFlow(login_url, state_token);
    setOAuthPopupAbort(() => abort);
    try {
      const { code, state } = await promise;
      await oAuthCallback(code, state);
      if (onSuccess) onSuccess();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setOAuthError(message);
      if (onError) onError(message);
    } finally {
      setOAuth2FlowInProgress(false);
      setOAuthPopupAbort(null);
    }
  }

  function handleCloseModal() {
    if (oAuthPopupAbort) oAuthPopupAbort();
    if (onCancel) onCancel();
  }

  return (
    <>
      <Button
        size="small"
        onClick={handleOAuthLogin}
        disabled={disabled || isOAuth2FlowInProgress}
      >
        {buttonText || `Sign in with ${providerName} account`}
      </Button>
      <OAuthFlowWaitingModal
        open={isOAuth2FlowInProgress}
        onClose={handleCloseModal}
        providerName={providerName}
      />
      {oAuthError && (
        <div className="mt-2 text-red-500">Error: {oAuthError}</div>
      )}
    </>
  );
}
