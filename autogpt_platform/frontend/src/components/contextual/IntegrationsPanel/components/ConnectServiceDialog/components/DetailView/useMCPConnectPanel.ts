"use client";

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  postV2DiscoverAvailableToolsOnAnMcpServer,
  postV2StoreABearerTokenForAnMcpServer,
} from "@/app/api/__generated__/endpoints/mcp/mcp";
import { useGetV1ListCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { useMCPAuthScheme } from "@/components/contextual/MCPAuthSchemeField/useMCPAuthScheme";
import {
  prepareMCPAuthCredential,
  validateMCPAuthCredential,
  type MCPAuthScheme,
} from "@/lib/mcp-auth";
import { getAPIResponseError, getErrorMessage } from "@/lib/mcp-errors";
import { mcpServerIdentity, normalizeMcpUrl } from "@/lib/mcp-url";
import { OAUTH_ERROR_FLOW_CANCELED } from "@/lib/oauth-popup";
import { invalidateConnectionQueries } from "@/lib/react-query/invalidateConnections";
import { connectMCPOAuth } from "./mcpOAuth";

interface Args {
  onSuccess: (credential?: CredentialsMetaResponse) => void;
  initialServerURL: string;
  initialAuthMode: string;
}

export function useMCPConnectPanel({
  onSuccess,
  initialServerURL,
  initialAuthMode,
}: Args) {
  const queryClient = useQueryClient();
  const { data: savedCredentials } = useGetV1ListCredentials({
    query: {
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });
  const [serverURL, setServerURL] = useState(initialServerURL);
  const [token, setToken] = useState("");
  const [phase, setPhase] = useState(
    initialAuthMode === "token" ? "manual-token" : "form",
  );
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isWaitingForOAuth, setIsWaitingForOAuth] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const oauthAbortRef = useRef<((reason?: string) => void) | null>(null);
  useEffect(() => () => oauthAbortRef.current?.(), []);

  const trimmedURL = serverURL.trim();
  const trimmedToken = token.trim();
  const canConnect = isValidHttpURL(trimmedURL) && !isSubmitting;
  const canSubmitToken = canConnect && trimmedToken.length > 0;
  const saved = Array.isArray(savedCredentials)
    ? savedCredentials.find(
        (credential) =>
          credential.provider === "mcp" &&
          typeof credential.host === "string" &&
          normalizeMcpUrl(credential.host) === normalizeMcpUrl(trimmedURL),
      )
    : null;
  const savedAuthScheme: MCPAuthScheme =
    saved?.mcp_auth_scheme === "basic" ? "basic" : "bearer";
  const {
    scheme: authScheme,
    selectScheme,
    detectSchemeFrom,
    resetScheme,
  } = useMCPAuthScheme(savedAuthScheme, token);

  async function handleConnect() {
    if (!canConnect) return;
    setError(null);
    setIsSubmitting(true);
    oauthAbortRef.current?.();
    try {
      const credential = await connectMCPOAuth({
        serverURL: trimmedURL,
        onPopup: (abort) => {
          oauthAbortRef.current = abort;
          setIsWaitingForOAuth(abort !== null);
        },
      });
      if (!credential) {
        setPhase("manual-token");
        setError(
          "This server doesn't support OAuth sign-in. Choose how its API credential should be sent.",
        );
        return;
      }
      await invalidateConnectionQueries(queryClient);
      onSuccess(credential);
    } catch (error) {
      const message = getErrorMessage(error);
      if (message === OAUTH_ERROR_FLOW_CANCELED) return;
      setError(
        message === "OAuth flow timed out"
          ? "OAuth sign-in timed out. Please try again."
          : message,
      );
    } finally {
      setIsSubmitting(false);
      setIsWaitingForOAuth(false);
      oauthAbortRef.current = null;
    }
  }

  async function handleSubmitToken() {
    if (!canSubmitToken) return;
    const invalid = validateMCPAuthCredential(trimmedToken, authScheme);
    if (invalid) {
      setError(invalid);
      return;
    }
    setError(null);
    setIsSubmitting(true);
    try {
      const authValue = prepareMCPAuthCredential(trimmedToken, authScheme);
      const probe = await postV2DiscoverAvailableToolsOnAnMcpServer({
        server_url: trimmedURL,
        auth_token: authValue,
      });
      if (probe.status !== 200)
        throw getAPIResponseError(probe.status, probe.data);
      const stored = await postV2StoreABearerTokenForAnMcpServer({
        server_url: trimmedURL,
        token: authValue,
      });
      if (stored.status !== 200)
        throw getAPIResponseError(stored.status, stored.data);
      await invalidateConnectionQueries(queryClient);
      onSuccess(stored.data);
    } catch (error) {
      setError(getErrorMessage(error));
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleSwitchToOAuth() {
    setPhase("form");
    setToken("");
    resetScheme();
    setError(null);
  }

  function handleSwitchToToken() {
    if (isSubmitting && !oauthAbortRef.current) return;
    oauthAbortRef.current?.();
    setPhase("manual-token");
    setError(null);
  }

  function handleServerURLChange(nextURL: string) {
    const changed = mcpServerIdentity(serverURL) !== mcpServerIdentity(nextURL);
    setServerURL(nextURL);
    if (!changed) return;
    setToken("");
    resetScheme();
    setPhase(initialAuthMode === "token" ? "manual-token" : "form");
    setError(null);
  }

  function handleTokenChange(value: string) {
    setToken(value);
    detectSchemeFrom(value);
  }

  return {
    serverURL,
    token,
    phase,
    isSubmitting,
    error,
    authScheme,
    selectScheme,
    canConnect,
    canSubmitToken,
    canSwitchToToken: !isSubmitting || isWaitingForOAuth,
    handleConnect,
    handleSubmitToken,
    handleSwitchToOAuth,
    handleSwitchToToken,
    handleServerURLChange,
    handleTokenChange,
  };
}

function isValidHttpURL(value: string) {
  try {
    const url = new URL(value);
    return url.protocol === "https:" || url.protocol === "http:";
  } catch {
    return false;
  }
}
