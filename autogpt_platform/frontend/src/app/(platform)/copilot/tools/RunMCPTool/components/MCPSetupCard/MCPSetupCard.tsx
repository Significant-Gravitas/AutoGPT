"use client";

import {
  postV2ExchangeOauthCodeForMcpTokens,
  postV2InitiateOauthLoginForAnMcpServer,
} from "@/app/api/__generated__/endpoints/mcp/mcp";
import { customMutator } from "@/app/api/mutators/custom-mutator";
import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import { Button } from "@/components/atoms/Button/Button";
import { openOAuthPopup } from "@/lib/oauth-popup";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import { useContext, useRef, useState } from "react";
import { useCopilotChatActions } from "../../../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { ContentMessage } from "../../../../components/ToolAccordion/AccordionContent";
import { serverHost } from "../../helpers";

interface Props {
  output: SetupRequirementsResponse;
  /**
   * Message sent to the chat after a successful connection.
   * Defaults to a generic "credentials connected, please retry" message.
   */
  retryInstruction?: string;
}

/**
 * Credential setup card for MCP servers — replaces the generic
 * SetupRequirementsCard (which uses the standard integration OAuth route)
 * with the MCP-specific OAuth flow (`/api/v2/mcp/oauth/*`).
 *
 * OAuth flow: initiate login → popup → exchange code for tokens.
 * Fallback: if the server doesn't support MCP OAuth (400), prompts the user
 * to provide an API token manually.
 */
export function MCPSetupCard({ output, retryInstruction }: Props) {
  const { onSend } = useCopilotChatActions();
  const allProviders = useContext(CredentialsProvidersContext);

  // setup_info.agent_id is set to the server_url in the backend
  const serverUrl = output.setup_info.agent_id;
  const host = serverHost(serverUrl);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showManualToken, setShowManualToken] = useState(false);
  const [manualToken, setManualToken] = useState("");
  const [connected, setConnected] = useState(false);
  const oauthAbortRef = useRef<(() => void) | null>(null);

  async function handleConnect() {
    setError(null);
    setLoading(true);
    oauthAbortRef.current?.();

    try {
      const loginRes = await postV2InitiateOauthLoginForAnMcpServer({
        server_url: serverUrl,
      });
      if (loginRes.status !== 200) throw loginRes.data;
      const { login_url, state_token } = loginRes.data as {
        login_url: string;
        state_token: string;
      };

      const { promise, cleanup } = openOAuthPopup(login_url, {
        stateToken: state_token,
        useCrossOriginListeners: true,
      });
      oauthAbortRef.current = cleanup.abort;

      const result = await promise;

      const mcpProvider = allProviders?.["mcp"];
      if (mcpProvider) {
        await mcpProvider.mcpOAuthCallback(result.code, state_token);
      } else {
        const cbRes = await postV2ExchangeOauthCodeForMcpTokens({
          code: result.code,
          state_token,
        });
        if (cbRes.status !== 200) throw cbRes.data;
      }

      setConnected(true);
      onSend(
        retryInstruction ??
          "I've connected the MCP server credentials. Please retry.",
      );
    } catch (e: unknown) {
      const err = e as Record<string, unknown>;
      if (err?.status === 400) {
        setShowManualToken(true);
        setError(
          "This server does not support OAuth sign-in. Please enter an API token manually.",
        );
      } else if (
        typeof err?.message === "string" &&
        err.message === "OAuth flow timed out"
      ) {
        setError("OAuth sign-in timed out. Please try again.");
      } else {
        const msg =
          (typeof err?.message === "string" ? err.message : null) ||
          (typeof err?.detail === "string" ? err.detail : null) ||
          "Failed to complete sign-in. Please try again.";
        setError(msg);
      }
    } finally {
      setLoading(false);
      oauthAbortRef.current = null;
    }
  }

  async function handleManualToken() {
    const token = manualToken.trim();
    if (!token) return;
    setLoading(true);
    setError(null);
    try {
      const res = await customMutator<{
        data: unknown;
        status: number;
        headers: Headers;
      }>("/v2/mcp/token", {
        method: "POST",
        body: JSON.stringify({ server_url: serverUrl, token }),
      });
      if (res.status !== 200) throw new Error("Failed to store token");
      setConnected(true);
      onSend(
        retryInstruction ??
          "I've connected the MCP server credentials. Please retry.",
      );
    } catch (e: unknown) {
      const err = e as Record<string, unknown>;
      setError(
        (typeof err?.message === "string" ? err.message : null) ||
          "Failed to save token. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  }

  if (connected) {
    return (
      <div className="mt-2 rounded-lg border border-green-200 bg-green-50 px-3 py-2 text-sm text-green-700">
        Connected to {host}!
      </div>
    );
  }

  return (
    <div className="mt-2 grid gap-2">
      <ContentMessage>{output.message}</ContentMessage>

      <div className="rounded-2xl border bg-background p-4">
        <Button
          variant="primary"
          size="small"
          onClick={handleConnect}
          disabled={loading}
        >
          {loading ? "Connecting…" : `Connect to ${host}`}
        </Button>

        {error && <p className="mt-2 text-sm text-red-600">{error}</p>}

        {showManualToken && (
          <div className="mt-3 flex gap-2">
            <input
              type="password"
              placeholder="Paste API token"
              value={manualToken}
              onChange={(e) => setManualToken(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleManualToken()}
              className="flex-1 rounded border px-2 py-1 text-sm"
            />
            <Button
              variant="secondary"
              size="small"
              onClick={handleManualToken}
              disabled={!manualToken.trim()}
            >
              Use Token
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
