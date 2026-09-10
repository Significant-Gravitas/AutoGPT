"use client";

import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { MCPAuthSchemeField } from "@/components/contextual/MCPAuthSchemeField/MCPAuthSchemeField";
import {
  mcpAuthTokenHint,
  mcpAuthTokenLabel,
  mcpAuthTokenPlaceholder,
} from "@/components/contextual/MCPAuthSchemeField/helpers";
import { useMCPConnectPanel } from "./useMCPConnectPanel";

interface Props {
  onSuccess: (credential?: CredentialsMetaResponse) => void;
  initialServerURL?: string;
  lockServerURL?: boolean;
  initialAuthMode?: "oauth" | "token" | "none" | "unknown";
}

export function McpConnectPanel({
  onSuccess,
  initialServerURL = "",
  lockServerURL = false,
  initialAuthMode = "unknown",
}: Props) {
  const state = useMCPConnectPanel({
    onSuccess,
    initialServerURL,
    initialAuthMode,
  });

  return (
    <div className="flex flex-col gap-4">
      {!lockServerURL && (
        <Text variant="body" className="text-zinc-600">
          Enter the URL of your MCP server. We&apos;ll try OAuth first and fall
          back to a manual API credential if the server doesn&apos;t support
          OAuth.
        </Text>
      )}
      <Input
        id="mcp-server-url"
        label="Server URL"
        type="url"
        placeholder="https://mcp.example.com"
        value={state.serverURL}
        onChange={(e) => state.handleServerURLChange(e.target.value)}
        disabled={state.isSubmitting}
        readOnly={lockServerURL}
        autoFocus={!lockServerURL}
      />
      {state.phase === "manual-token" && (
        <>
          <MCPAuthSchemeField
            value={state.authScheme}
            onChange={state.selectScheme}
            disabled={state.isSubmitting}
            className="flex flex-col gap-1"
            labelClassName="text-sm font-medium text-zinc-700"
            selectClassName="rounded-lg border border-zinc-300 bg-white px-3 py-2 text-zinc-900"
          />
          <Input
            id="mcp-auth-token"
            label={mcpAuthTokenLabel(state.authScheme)}
            type="password"
            placeholder={mcpAuthTokenPlaceholder(state.authScheme)}
            value={state.token}
            onChange={(e) => state.handleTokenChange(e.target.value)}
            disabled={state.isSubmitting}
            hint={mcpAuthTokenHint(state.authScheme)}
          />
        </>
      )}
      {state.error && (
        <div
          role="alert"
          aria-live="polite"
          className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700"
        >
          {state.error}
        </div>
      )}
      <div className="flex items-center justify-end gap-2">
        {state.phase === "manual-token" && (
          <Button
            variant="secondary"
            size="small"
            onClick={state.handleSwitchToOAuth}
            disabled={state.isSubmitting}
          >
            Try OAuth
          </Button>
        )}
        <Button
          variant="primary"
          size="small"
          onClick={
            state.phase === "form"
              ? state.handleConnect
              : state.handleSubmitToken
          }
          disabled={
            state.phase === "form" ? !state.canConnect : !state.canSubmitToken
          }
          loading={state.isSubmitting}
        >
          {state.phase === "form" ? "Connect" : "Save token"}
        </Button>
      </div>
    </div>
  );
}
