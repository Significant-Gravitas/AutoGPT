"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ProviderAvatar } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/ProviderAvatar";
import { MCPAuthSchemeField } from "@/components/contextual/MCPAuthSchemeField/MCPAuthSchemeField";
import {
  mcpAuthTokenHint,
  mcpAuthTokenLabel,
} from "@/components/contextual/MCPAuthSchemeField/helpers";
import { useMCPAuthScheme } from "@/components/contextual/MCPAuthSchemeField/useMCPAuthScheme";
import { prepareMCPAuthCredential } from "@/lib/mcp-auth";
import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { useId, useState } from "react";
import type { McpConnectorRequest } from "./helpers";

function hostOf(serverUrl: string): string | null {
  try {
    return new URL(serverUrl).hostname;
  } catch {
    return null;
  }
}

/** MCP server row inside the connectors table. Visually mirrors
 *  ConnectorRow but connects through the MCP OAuth/token flow the hidden
 *  MCPSetupCard drives via the request callbacks. */
export function McpConnectorRow({ request }: { request: McpConnectorRequest }) {
  const hintId = useId();
  const [token, setToken] = useState("");
  const {
    scheme: authScheme,
    selectScheme,
    detectSchemeFrom,
  } = useMCPAuthScheme(request.authScheme, token);
  const host = hostOf(request.serverUrl);
  // "mcp.notion.com" → "notion" so existing /integrations/*.png icons
  // resolve; unknown services fall back to the avatar's initial.
  const iconId = (host ?? request.service)
    .replace(/^mcp\./, "")
    .split(".")[0]
    .toLowerCase();

  function submitCredential() {
    const credential = prepareMCPAuthCredential(token, authScheme);
    if (credential) request.onUseToken(credential);
  }

  return (
    <div className="flex flex-col">
      <div className="flex items-center gap-3 px-4 py-3">
        <span className="flex size-10 shrink-0 items-center justify-center overflow-hidden rounded-2xl border border-zinc-100 bg-white p-1.5">
          <ProviderAvatar id={iconId} name={request.service} />
        </span>

        <div className="flex min-w-0 flex-1 flex-col">
          <span className="truncate text-sm font-medium text-zinc-900">
            {request.service}
          </span>
          {host && (
            <span className="truncate text-sm text-zinc-500">{host}</span>
          )}
        </div>

        {request.connected ? (
          <span className="flex shrink-0 items-center gap-1.5 text-sm font-medium text-zinc-500">
            <Icon icon={CheckmarkCircle02Icon} size={16} />
            Connected
          </span>
        ) : (
          <Button
            variant="primary"
            size="small"
            className="shrink-0"
            disabled={request.loading}
            onClick={request.onConnect}
          >
            {request.loading ? "Connecting…" : "Connect"}
          </Button>
        )}
      </div>

      {request.error && !request.connected && (
        <div className="mx-4 mb-3 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {request.error}
        </div>
      )}

      {request.showManualToken && !request.connected && (
        <div className="mx-4 mb-3 grid gap-2">
          <MCPAuthSchemeField
            value={authScheme}
            onChange={selectScheme}
            disabled={request.loading}
            nameSuffix={request.service}
            className="grid gap-1"
            labelClassName="text-xs font-medium text-zinc-700"
            selectClassName="rounded-xl bg-zinc-50 px-3 py-2 text-sm text-zinc-800 ring-1 ring-zinc-100"
          />
          <p className="text-xs text-zinc-500">
            {mcpAuthTokenHint(authScheme)}
          </p>
          <div className="flex gap-2">
            <input
              type="password"
              aria-describedby={hintId}
              aria-label={`${mcpAuthTokenLabel(authScheme)} for ${request.service}`}
              placeholder="Paste API token"
              value={token}
              onChange={(e) => {
                const nextToken = e.target.value;
                setToken(nextToken);
                detectSchemeFrom(nextToken);
              }}
              onKeyDown={(e) =>
                e.key === "Enter" &&
                !request.loading &&
                token.trim() &&
                submitCredential()
              }
              className="flex-1 rounded-2xl bg-zinc-50 px-3 py-2 text-sm text-zinc-800 ring-1 ring-zinc-100 transition-shadow placeholder:text-zinc-400 focus:outline-none focus:ring-zinc-300"
            />
            <Button
              variant="secondary"
              size="small"
              disabled={request.loading || !token.trim()}
              onClick={submitCredential}
            >
              Use Token
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
