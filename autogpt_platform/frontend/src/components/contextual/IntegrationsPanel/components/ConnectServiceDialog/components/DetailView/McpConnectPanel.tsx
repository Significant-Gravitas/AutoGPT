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
import type { MCPAuthScheme } from "@/lib/mcp-auth";
import { MCPServerURLField } from "./MCPServerURLField";
import { MCPWriteAccessField } from "./MCPWriteAccessField";

interface Props {
  onSuccess: (credential?: CredentialsMetaResponse) => void;
  initialServerURL?: string;
  lockServerURL?: boolean;
  initialAuthMode?: "oauth" | "token" | "none" | "unknown";
  allowedAuthMethods?: ("oauth" | MCPAuthScheme)[];
  oauthScopes?: string[] | null;
  oauthWriteScopes?: string[];
  serverURLOptions?: { label: string; url: string }[];
}

export function McpConnectPanel({
  onSuccess,
  initialServerURL = "",
  lockServerURL = false,
  initialAuthMode = "unknown",
  allowedAuthMethods,
  oauthScopes,
  oauthWriteScopes = [],
  serverURLOptions,
}: Props) {
  const state = useMCPConnectPanel({
    onSuccess,
    initialServerURL,
    initialAuthMode,
    allowedAuthMethods,
    oauthScopes,
    oauthWriteScopes,
  });

  return (
    <div className="flex flex-col gap-4">
      {!lockServerURL && (
        <Text variant="body" className="text-zinc-600">
          Enter the server URL from the service&apos;s setup instructions.
        </Text>
      )}
      <MCPServerURLField
        serverURL={state.serverURL}
        onChange={state.handleServerURLChange}
        disabled={state.isSubmitting}
        readOnly={lockServerURL}
        options={serverURLOptions}
      />
      {state.phase === "form" && oauthWriteScopes.length > 0 && (
        <MCPWriteAccessField
          checked={state.allowChanges}
          onChange={state.setAllowChanges}
          disabled={state.isSubmitting}
        />
      )}
      {state.phase === "manual-token" && (
        <>
          <Text variant="small" className="text-zinc-600">
            {allowedAuthMethods
              ? "Use the credential described in the setup instructions above."
              : "Use an API credential only if this server supports it. Follow the server's documentation for the correct authentication type."}
          </Text>
          {state.manualSchemes.length > 1 && (
            <MCPAuthSchemeField
              value={state.authScheme}
              onChange={state.selectScheme}
              disabled={state.isSubmitting}
              className="flex flex-col gap-1"
              labelClassName="text-sm font-medium text-zinc-700"
              selectClassName="rounded-lg border border-zinc-300 bg-white px-3 py-2 text-zinc-900"
            />
          )}
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
        {state.phase === "form" && state.manualSchemes.length > 0 && (
          <Button
            variant="secondary"
            size="small"
            onClick={state.handleSwitchToToken}
            disabled={!state.canSwitchToToken}
          >
            Use an API token instead
          </Button>
        )}
        {state.phase === "manual-token" && state.canUseOAuth && (
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
