"use client";

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { MCPAuthScheme } from "@/lib/mcp-auth";
import { getErrorMessage } from "@/lib/mcp-errors";
import { normalizeMcpUrl } from "@/lib/mcp-url";
import { OAUTH_ERROR_FLOW_CANCELED } from "@/lib/oauth-popup";
import { invalidateConnectionQueries } from "@/lib/react-query/invalidateConnections";
import { connectMCPOAuth } from "@/lib/mcp-oauth";
import { storeMCPToken } from "./storeMCPToken";
import { useMCPManualAuth } from "./useMCPManualAuth";

interface Args {
  onSuccess: (credential?: CredentialsMetaResponse) => void;
  initialServerURL: string;
  allowedAuthMethods?: ("oauth" | MCPAuthScheme)[];
  oauthScopes?: string[] | null;
  oauthWriteScopes?: string[];
}

export function useMCPConnectPanel({
  onSuccess,
  initialServerURL,
  allowedAuthMethods = ["oauth", "bearer", "basic"],
  oauthScopes,
  oauthWriteScopes = [],
}: Args) {
  const queryClient = useQueryClient();
  const [serverURL, setServerURL] = useState(initialServerURL);
  const initialPhase =
    allowedAuthMethods[0] === "oauth" ? "form" : "manual-token";
  const [phase, setPhase] = useState(initialPhase);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [allowChanges, setAllowChanges] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const activeRequest = useRef<AbortController | null>(null);
  useEffect(() => () => activeRequest.current?.abort(), []);

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
    if (
      !canConnect ||
      (activeRequest.current && !activeRequest.current.signal.aborted)
    )
      return;
    setError(null);
    setIsSubmitting(true);
    const signal = startRequest();
    try {
      const credential = await connectMCPOAuth({
        serverURL: trimmedURL,
        scopes:
          oauthScopes == null
            ? undefined
            : [
                ...new Set([
                  ...oauthScopes,
                  ...(allowChanges ? oauthWriteScopes : []),
                ]),
              ],
        signal,
      });
      signal.throwIfAborted();
      if (!credential || "reason" in credential) {
        const reason =
          credential && "reason" in credential ? credential.reason : undefined;
        // Only "this server has no OAuth" should push the user at the manual
        // token tab; every other 400 is its own problem and says so. The route
        // makes that call and sends `no_oauth`, because the same verdict comes
        // back under two quite different messages.
        const noOAuth =
          !credential || ("noOAuth" in credential && credential.noOAuth);
        if (noOAuth && manualSchemes.length) setPhase("manual-token");
        setError(
          reason ??
            (manualSchemes.length
              ? "This server doesn't support OAuth sign-in. Choose how its API credential should be sent."
              : "Sign-in is unavailable for this connection. Check its setup instructions and try again."),
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
      if (activeRequest.current?.signal === signal) {
        activeRequest.current = null;
        if (!signal.aborted) setIsSubmitting(false);
      }
    }
  }

  async function handleSubmitToken() {
    if (
      !canSubmitToken ||
      (activeRequest.current && !activeRequest.current.signal.aborted)
    )
      return;
    const invalid = manual.validateToken();
    if (invalid) {
      setError(invalid);
      return;
    }
    setError(null);
    setIsSubmitting(true);
    const signal = startRequest();
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
      if (activeRequest.current?.signal === signal) {
        activeRequest.current = null;
        if (!signal.aborted) setIsSubmitting(false);
      }
    }
  }

  function handleSwitchToOAuth() {
    if (!canUseOAuth) return;
    activeRequest.current?.abort();
    setIsSubmitting(false);
    setPhase("form");
    manual.reset();
    setError(null);
  }

  function handleSwitchToToken() {
    if (!manualSchemes.length) return;
    activeRequest.current?.abort();
    setIsSubmitting(false);
    setPhase("manual-token");
    setError(null);
  }

  function handleServerURLChange(nextURL: string) {
    // Compare the whole URL, not just the origin: two tenants on one host
    // differ only by path, and keeping the typed token across that change
    // would submit one tenant's secret to another.
    const changed = normalizeMcpUrl(serverURL) !== normalizeMcpUrl(nextURL);
    setServerURL(nextURL);
    if (!changed) return;
    activeRequest.current?.abort();
    setIsSubmitting(false);
    manual.reset();
    setAllowChanges(false);
    setPhase(initialPhase);
    setError(null);
  }

  function startRequest() {
    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;
    return controller.signal;
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
