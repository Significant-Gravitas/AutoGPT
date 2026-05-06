"use client";

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";

import {
  postV2ExchangeOauthCodeForMcpTokens,
  postV2InitiateOauthLoginForAnMcpServer,
  postV2StoreABearerTokenForAnMcpServer,
} from "@/app/api/__generated__/endpoints/mcp/mcp";
import { getGetV1ListCredentialsQueryKey } from "@/app/api/__generated__/endpoints/integrations/integrations";
import type { MCPOAuthLoginResponse } from "@/app/api/__generated__/models/mCPOAuthLoginResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { openOAuthPopup } from "@/lib/oauth-popup";

interface Props {
  onSuccess: () => void;
}

type Phase = "form" | "manual-token";

export function McpConnectPanel({ onSuccess }: Props) {
  const queryClient = useQueryClient();
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

  async function invalidateCredentials() {
    await queryClient.invalidateQueries({
      queryKey: getGetV1ListCredentialsQueryKey(),
    });
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
      } catch (e: unknown) {
        if (getErrorStatus(e) === 400) {
          setPhase("manual-token");
          setError(
            "This server doesn't support OAuth sign-in. Paste a bearer token instead.",
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

      await postV2ExchangeOauthCodeForMcpTokens({
        code: result.code,
        state_token,
      });

      await invalidateCredentials();
      onSuccess();
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
    setError(null);
    setIsSubmitting(true);

    try {
      await postV2StoreABearerTokenForAnMcpServer({
        server_url: trimmedUrl,
        token: trimmedToken,
      });

      await invalidateCredentials();
      onSuccess();
    } catch (e: unknown) {
      setError(getErrorMessage(e));
    } finally {
      setIsSubmitting(false);
    }
  }

  function handleSwitchToOAuth() {
    setPhase("form");
    setToken("");
    setError(null);
  }

  return (
    <div className="flex flex-col gap-4">
      <Text variant="body" className="text-zinc-600">
        Enter the URL of your MCP server. We&apos;ll try OAuth first and fall
        back to a bearer token if the server doesn&apos;t support OAuth.
      </Text>

      <Input
        id="mcp-server-url"
        label="Server URL"
        type="url"
        placeholder="https://mcp.example.com"
        value={serverUrl}
        onChange={(e) => setServerUrl(e.target.value)}
        disabled={isSubmitting}
        autoFocus
      />

      {phase === "manual-token" ? (
        <Input
          id="mcp-bearer-token"
          label="Bearer token"
          type="password"
          placeholder="Paste API token"
          value={token}
          onChange={(e) => setToken(e.target.value)}
          disabled={isSubmitting}
          hint="Used as a Bearer Authorization header when calling tools."
        />
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

function getErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === "object" && error !== null) {
    const detail = (error as { detail?: unknown }).detail;
    if (typeof detail === "string") return detail;
    const message = (error as { message?: unknown }).message;
    if (typeof message === "string") return message;
  }
  return "Something went wrong. Please try again.";
}

function getErrorStatus(error: unknown): number | null {
  if (typeof error === "object" && error !== null) {
    const status = (error as { status?: unknown }).status;
    if (typeof status === "number") return status;
  }
  return null;
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
