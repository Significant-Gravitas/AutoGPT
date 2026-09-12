"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { MCPAuthScheme } from "@/lib/mcp-auth";
import { getErrorMessage } from "@/lib/mcp-errors";
import { mcpServerIdentity } from "@/lib/mcp-url";
import { OAUTH_ERROR_FLOW_CANCELED } from "@/lib/oauth-popup";
import { invalidateConnectionQueries } from "@/lib/react-query/invalidateConnections";
import { connectMCPOAuth } from "./mcpOAuth";
import { mcpOAuthScopes } from "./mcpPresetHelpers";
import { storeMCPToken } from "./storeMCPToken";
import { useMCPManualAuth } from "./useMCPManualAuth";
import { useMCPRequest } from "./useMCPRequest";

interface Args {
  onSuccess: (credential?: CredentialsMetaResponse) => void;
  initialServerURL: string;
  initialAuthMode: string;
  allowedAuthMethods?: ("oauth" | MCPAuthScheme)[];
  oauthScopes?: string[] | null;
  oauthWriteScopes?: string[];
}

export function useMCPConnectPanel({
  onSuccess,
  initialServerURL,
  initialAuthMode,
  allowedAuthMethods = ["oauth", "bearer", "basic"],
  oauthScopes,
  oauthWriteScopes = [],
}: Args) {
  const queryClient = useQueryClient();
  const [serverURL, setServerURL] = useState(initialServerURL);
  const initialPhase = initialAuthMode === "token" ? "manual-token" : "form";
  const [phase, setPhase] = useState(initialPhase);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [allowChanges, setAllowChanges] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const request = useMCPRequest();

  const manualSchemes = allowedAuthMethods.filter(
    (method): method is MCPAuthScheme => method !== "oauth",
  );
  const canUseOAuth = allowedAuthMethods.includes("oauth");
  const manual = useMCPManualAuth(serverURL, manualSchemes);
  const trimmedURL = serverURL.trim();
  const validURL = isValidHttpURL(trimmedURL);
  const canConnect = validURL && !isSubmitting && canUseOAuth;
  const canSubmitToken =
    validURL &&
    !isSubmitting &&
    manualSchemes.length > 0 &&
    manual.token.trim().length > 0;

  async function handleConnect() {
    if (!canConnect) return;
    setError(null);
    setIsSubmitting(true);
    const signal = request.start();
    try {
      const credential = await connectMCPOAuth({
        serverURL: trimmedURL,
        scopes: mcpOAuthScopes(oauthScopes, oauthWriteScopes, allowChanges),
        signal,
      });
      signal.throwIfAborted();
      if (!credential) {
        if (manualSchemes.length) setPhase("manual-token");
        setError(
          manualSchemes.length
            ? "This server doesn't support OAuth sign-in. Choose how its API credential should be sent."
            : "Sign-in is unavailable for this connection. Check its setup instructions and try again.",
        );
        return;
      }
      await invalidateConnectionQueries(queryClient);
      signal.throwIfAborted();
      onSuccess(credential);
    } catch (error) {
      if (signal.aborted) return;
      const message = getErrorMessage(error);
      if (message === OAUTH_ERROR_FLOW_CANCELED) return;
      setError(
        message === "OAuth flow timed out"
          ? "OAuth sign-in timed out. Please try again."
          : message,
      );
    } finally {
      if (!signal.aborted) setIsSubmitting(false);
    }
  }

  async function handleSubmitToken() {
    if (!canSubmitToken) return;
    const invalid = manual.validateToken();
    if (invalid) {
      setError(invalid);
      return;
    }
    setError(null);
    setIsSubmitting(true);
    const signal = request.start();
    try {
      const credential = await storeMCPToken(
        trimmedURL,
        manual.token.trim(),
        manual.scheme,
        signal,
      );
      signal.throwIfAborted();
      await invalidateConnectionQueries(queryClient);
      signal.throwIfAborted();
      onSuccess(credential);
    } catch (error) {
      if (signal.aborted) return;
      setError(getErrorMessage(error));
    } finally {
      if (!signal.aborted) setIsSubmitting(false);
    }
  }

  function handleSwitchToOAuth() {
    if (!canUseOAuth) return;
    request.cancel();
    setIsSubmitting(false);
    setPhase("form");
    manual.reset();
    setError(null);
  }

  function handleSwitchToToken() {
    if (!manualSchemes.length) return;
    request.cancel();
    setIsSubmitting(false);
    setPhase("manual-token");
    setError(null);
  }

  function handleServerURLChange(nextURL: string) {
    const changed = mcpServerIdentity(serverURL) !== mcpServerIdentity(nextURL);
    setServerURL(nextURL);
    if (!changed) return;
    request.cancel();
    setIsSubmitting(false);
    manual.reset();
    setAllowChanges(false);
    setPhase(initialPhase);
    setError(null);
  }

  return {
    serverURL,
    token: manual.token,
    phase,
    isSubmitting,
    error,
    allowChanges,
    setAllowChanges,
    authScheme: manual.scheme,
    selectScheme: manual.selectScheme,
    manualSchemes,
    canUseOAuth,
    canConnect,
    canSubmitToken,
    canSwitchToToken: manualSchemes.length > 0,
    handleConnect,
    handleSubmitToken,
    handleSwitchToOAuth,
    handleSwitchToToken,
    handleServerURLChange,
    handleTokenChange: manual.handleTokenChange,
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
