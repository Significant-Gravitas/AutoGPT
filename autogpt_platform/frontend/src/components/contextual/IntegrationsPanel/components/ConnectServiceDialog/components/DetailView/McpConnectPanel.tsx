"use client";

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import {
  postV2DiscoverAvailableToolsOnAnMcpServer,
  postV2ExchangeOauthCodeForMcpTokens,
  postV2InitiateOauthLoginForAnMcpServer,
  postV2StoreABearerTokenForAnMcpServer,
} from "@/app/api/__generated__/endpoints/mcp/mcp";
import { useGetV1ListCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { MCPOAuthLoginResponse } from "@/app/api/__generated__/models/mCPOAuthLoginResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { MCPAuthSchemeField } from "@/components/contextual/MCPAuthSchemeField/MCPAuthSchemeField";
import {
  mcpAuthTokenHint,
  mcpAuthTokenLabel,
  mcpAuthTokenPlaceholder,
} from "@/components/contextual/MCPAuthSchemeField/helpers";
import { useMCPAuthScheme } from "@/components/contextual/MCPAuthSchemeField/useMCPAuthScheme";
import {
  prepareMCPAuthCredential,
  validateMCPAuthCredential,
  type MCPAuthScheme,
} from "@/lib/mcp-auth";
import {
  getAPIResponseError,
  getErrorMessage,
  getErrorStatus,
} from "@/lib/mcp-errors";
import { mcpServerIdentity, normalizeMcpUrl } from "@/lib/mcp-url";
import { openOAuthPopup } from "@/lib/oauth-popup";
import { invalidateConnectionQueries } from "@/lib/react-query/invalidateConnections";

interface Props {
  onSuccess: (credential?: CredentialsMetaResponse) => void;
}

type Phase = "form" | "manual-token";

export function McpConnectPanel({ onSuccess }: Props) {
  const queryClient = useQueryClient();
  const { data: savedCredentials } = useGetV1ListCredentials({
    query: {
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });
  const [serverUrl, setServerUrl] = useState("");
  const [token, setToken] = useState("");
  const [phase, setPhase] = useState<Phase>("form");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const oauthAbortRef = useRef<((reason?: string) => void) | null>(null);

  useEffect(() => () => oauthAbortRef.current?.(), []);

  const trimmedUrl = serverUrl.trim();
  const trimmedToken = token.trim();
  const isUrlValid = isValidHttpUrl(trimmedUrl);
  const canConnect = isUrlValid && !isSubmitting;
  const canSubmitToken = isUrlValid && trimmedToken.length > 0 && !isSubmitting;
  const savedCredential = Array.isArray(savedCredentials)
    ? savedCredentials.find(
        (credential) =>
          credential.provider === "mcp" &&
          typeof credential.host === "string" &&
          normalizeMcpUrl(credential.host) === normalizeMcpUrl(trimmedUrl),
      )
    : null;
  const savedAuthScheme: MCPAuthScheme =
    savedCredential?.mcp_auth_scheme === "basic" ? "basic" : "bearer";
  const {
    scheme: authScheme,
    selectScheme,
    detectSchemeFrom,
    resetScheme,
  } = useMCPAuthScheme(savedAuthScheme, token);

  async function invalidateCredentials() {
    await invalidateConnectionQueries(queryClient);
  }

  async function handleConnect() {
    if (!canConnect) return;
    setError(null);
    setIsSubmitting(true);
    oauthAbortRef.current?.();

    try {
      // Only a 400 from the *initiate* call means "server doesn't support
      // OAuth" — fall back to manual-token for that. A 400 from anywhere else
      // (popup callback, token exchange) is a real error and should surface
      // as such instead of forcing the manual-token UI.
      let loginRes: Awaited<
        ReturnType<typeof postV2InitiateOauthLoginForAnMcpServer>
      >;
      try {
        loginRes = await postV2InitiateOauthLoginForAnMcpServer({
          server_url: trimmedUrl,
        });
        if (loginRes.status !== 200) {
          throw getAPIResponseError(loginRes.status, loginRes.data);
        }
      } catch (e: unknown) {
        if (getErrorStatus(e) === 400) {
          setPhase("manual-token");
          setError(
            "This server doesn't support OAuth sign-in. Choose how its API credential should be sent.",
          );
          return;
        }
        throw e;
      }

      const { login_url, state_token } = loginRes.data as MCPOAuthLoginResponse;

      const { promise, cleanup } = openOAuthPopup(login_url, {
        stateToken: state_token,
        useCrossOriginListeners: true,
      });
      oauthAbortRef.current = cleanup.abort;

      const result = await promise;

      const exchanged = await postV2ExchangeOauthCodeForMcpTokens({
        code: result.code,
        state_token,
      });
      if (exchanged.status !== 200) {
        throw getAPIResponseError(exchanged.status, exchanged.data);
      }

      await invalidateCredentials();
      onSuccess(exchanged.data);
    } catch (e: unknown) {
      const message = getErrorMessage(e);
      if (message === "OAuth flow timed out") {
        setError("OAuth sign-in timed out. Please try again.");
      } else {
        setError(message);
      }
    } finally {
      setIsSubmitting(false);
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

      // Probe before storing so a rejected credential never replaces a working one.
      const probe = await postV2DiscoverAvailableToolsOnAnMcpServer({
        server_url: trimmedUrl,
        auth_token: authValue,
      });
      if (probe.status !== 200) {
        throw getAPIResponseError(probe.status, probe.data);
      }

      const stored = await postV2StoreABearerTokenForAnMcpServer({
        server_url: trimmedUrl,
        token: authValue,
      });
      if (stored.status !== 200) {
        throw getAPIResponseError(stored.status, stored.data);
      }

      await invalidateCredentials();
      onSuccess(stored.data);
    } catch (e: unknown) {
      setError(getErrorMessage(e));
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

  function handleServerUrlChange(nextUrl: string) {
    const serverIdentityChanged =
      mcpServerIdentity(serverUrl) !== mcpServerIdentity(nextUrl);

    setServerUrl(nextUrl);
    if (!serverIdentityChanged) return;

    setToken("");
    resetScheme();
    setPhase("form");
    setError(null);
  }

  return (
    <div className="flex flex-col gap-4">
      <Text variant="body" className="text-zinc-600">
        Enter the URL of your MCP server. We&apos;ll try OAuth first and fall
        back to a manual API credential if the server doesn&apos;t support
        OAuth.
      </Text>

      <Input
        id="mcp-server-url"
        label="Server URL"
        type="url"
        placeholder="https://mcp.example.com"
        value={serverUrl}
        onChange={(e) => handleServerUrlChange(e.target.value)}
        disabled={isSubmitting}
        autoFocus
      />

      {phase === "manual-token" ? (
        <>
          <MCPAuthSchemeField
            value={authScheme}
            onChange={selectScheme}
            disabled={isSubmitting}
            className="flex flex-col gap-1"
            labelClassName="text-sm font-medium text-zinc-700"
            selectClassName="rounded-lg border border-zinc-300 bg-white px-3 py-2 text-zinc-900"
          />
          <Input
            id="mcp-auth-token"
            label={mcpAuthTokenLabel(authScheme)}
            type="password"
            placeholder={mcpAuthTokenPlaceholder(authScheme)}
            value={token}
            onChange={(e) => {
              const nextToken = e.target.value;
              setToken(nextToken);
              detectSchemeFrom(nextToken);
            }}
            disabled={isSubmitting}
            hint={mcpAuthTokenHint(authScheme)}
          />
        </>
      ) : null}

      {error ? (
        <div
          role="alert"
          aria-live="polite"
          className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700"
        >
          {error}
        </div>
      ) : null}

      <div className="flex items-center justify-end gap-2">
        {phase === "manual-token" ? (
          <Button
            variant="secondary"
            size="small"
            onClick={handleSwitchToOAuth}
            disabled={isSubmitting}
          >
            Try OAuth
          </Button>
        ) : null}
        {phase === "form" ? (
          <Button
            variant="primary"
            size="small"
            onClick={handleConnect}
            disabled={!canConnect}
            loading={isSubmitting}
          >
            Connect
          </Button>
        ) : (
          <Button
            variant="primary"
            size="small"
            onClick={handleSubmitToken}
            disabled={!canSubmitToken}
            loading={isSubmitting}
          >
            Save token
          </Button>
        )}
      </div>
    </div>
  );
}

function isValidHttpUrl(value: string): boolean {
  if (!value) return false;
  try {
    const u = new URL(value);
    return u.protocol === "http:" || u.protocol === "https:";
  } catch {
    return false;
  }
}
