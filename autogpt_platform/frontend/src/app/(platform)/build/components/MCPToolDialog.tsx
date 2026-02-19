"use client";

import React, {
  useState,
  useCallback,
  useRef,
  useEffect,
  useContext,
} from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/__legacy__/ui/dialog";
import { Button } from "@/components/__legacy__/ui/button";
import { Input } from "@/components/__legacy__/ui/input";
import { Label } from "@/components/__legacy__/ui/label";
import { LoadingSpinner } from "@/components/__legacy__/ui/loading";
import { Badge } from "@/components/__legacy__/ui/badge";
import { ScrollArea } from "@/components/__legacy__/ui/scroll-area";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api";
import type { MCPToolResponse } from "@/app/api/__generated__/models/mCPToolResponse";
import {
  postV2DiscoverAvailableToolsOnAnMcpServer,
  postV2InitiateOauthLoginForAnMcpServer,
  postV2ExchangeOauthCodeForMcpTokens,
} from "@/app/api/__generated__/endpoints/mcp/mcp";
import { CaretDown } from "@phosphor-icons/react";
import { openOAuthPopup } from "@/lib/oauth-popup";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";

export type MCPToolDialogResult = {
  serverUrl: string;
  serverName: string | null;
  selectedTool: string;
  toolInputSchema: Record<string, any>;
  availableTools: Record<string, any>;
  /** Credentials meta from OAuth flow, null for public servers. */
  credentials: CredentialsMetaInput | null;
};

interface MCPToolDialogProps {
  open: boolean;
  onClose: () => void;
  onConfirm: (result: MCPToolDialogResult) => void;
}

type DialogStep = "url" | "tool";

export function MCPToolDialog({
  open,
  onClose,
  onConfirm,
}: MCPToolDialogProps) {
  const allProviders = useContext(CredentialsProvidersContext);

  const [step, setStep] = useState<DialogStep>("url");
  const [serverUrl, setServerUrl] = useState("");
  const [tools, setTools] = useState<MCPToolResponse[]>([]);
  const [serverName, setServerName] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [authRequired, setAuthRequired] = useState(false);
  const [oauthLoading, setOauthLoading] = useState(false);
  const [showManualToken, setShowManualToken] = useState(false);
  const [manualToken, setManualToken] = useState("");
  const [selectedTool, setSelectedTool] = useState<MCPToolResponse | null>(
    null,
  );
  const [credentials, setCredentials] = useState<CredentialsMetaInput | null>(
    null,
  );

  const startOAuthRef = useRef(false);
  const oauthAbortRef = useRef<((reason?: string) => void) | null>(null);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      oauthAbortRef.current?.();
    };
  }, []);

  const reset = useCallback(() => {
    oauthAbortRef.current?.();
    oauthAbortRef.current = null;
    setStep("url");
    setServerUrl("");
    setManualToken("");
    setTools([]);
    setServerName(null);
    setLoading(false);
    setError(null);
    setAuthRequired(false);
    setOauthLoading(false);
    setShowManualToken(false);
    setSelectedTool(null);
    setCredentials(null);
  }, []);

  const handleClose = useCallback(() => {
    reset();
    onClose();
  }, [reset, onClose]);

  const discoverTools = useCallback(async (url: string, authToken?: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await postV2DiscoverAvailableToolsOnAnMcpServer({
        server_url: url,
        auth_token: authToken || null,
      });
      if (response.status !== 200) throw response.data;
      setTools(response.data.tools);
      setServerName(response.data.server_name ?? null);
      setAuthRequired(false);
      setShowManualToken(false);
      setStep("tool");
    } catch (e: any) {
      if (e?.status === 401 || e?.status === 403) {
        setAuthRequired(true);
        setError(null);
        // Automatically start OAuth sign-in instead of requiring a second click
        setLoading(false);
        startOAuthRef.current = true;
        return;
      } else {
        const message =
          e?.message || e?.detail || "Failed to connect to MCP server";
        setError(
          typeof message === "string" ? message : JSON.stringify(message),
        );
      }
    } finally {
      setLoading(false);
    }
  }, []);

  const handleDiscoverTools = useCallback(() => {
    if (!serverUrl.trim()) return;
    discoverTools(serverUrl.trim(), manualToken.trim() || undefined);
  }, [serverUrl, manualToken, discoverTools]);

  const handleOAuthSignIn = useCallback(async () => {
    if (!serverUrl.trim()) return;
    setError(null);

    // Abort any previous OAuth flow
    oauthAbortRef.current?.();

    setOauthLoading(true);

    try {
      const loginResponse = await postV2InitiateOauthLoginForAnMcpServer({
        server_url: serverUrl.trim(),
      });
      if (loginResponse.status !== 200) throw loginResponse.data;
      const { login_url, state_token } = loginResponse.data;

      const { promise, cleanup } = openOAuthPopup(login_url, {
        stateToken: state_token,
        useCrossOriginListeners: true,
      });
      oauthAbortRef.current = cleanup.abort;

      const result = await promise;

      // Exchange code for tokens via the credentials provider (updates cache)
      setLoading(true);
      setOauthLoading(false);

      const mcpProvider = allProviders?.["mcp"];
      let callbackResult;
      if (mcpProvider) {
        callbackResult = await mcpProvider.mcpOAuthCallback(
          result.code,
          state_token,
        );
      } else {
        const cbResponse = await postV2ExchangeOauthCodeForMcpTokens({
          code: result.code,
          state_token,
        });
        if (cbResponse.status !== 200) throw cbResponse.data;
        callbackResult = cbResponse.data;
      }

      setCredentials({
        id: callbackResult.id,
        provider: callbackResult.provider,
        type: callbackResult.type,
        title: callbackResult.title,
      });
      setAuthRequired(false);

      // Discover tools now that we're authenticated
      const toolsResponse = await postV2DiscoverAvailableToolsOnAnMcpServer({
        server_url: serverUrl.trim(),
      });
      if (toolsResponse.status !== 200) throw toolsResponse.data;
      setTools(toolsResponse.data.tools);
      setServerName(toolsResponse.data.server_name ?? null);
      setStep("tool");
    } catch (e: any) {
      // If server doesn't support OAuth → show manual token entry
      if (e?.status === 400) {
        setShowManualToken(true);
        setError(
          "This server does not support OAuth sign-in. Please enter a token manually.",
        );
      } else if (e?.message === "OAuth flow timed out") {
        setError("OAuth sign-in timed out. Please try again.");
      } else {
        const status = e?.status;
        let message: string;
        if (status === 401 || status === 403) {
          message =
            "Authentication succeeded but the server still rejected the request. " +
            "The token audience may not match. Please try again.";
        } else {
          message = e?.message || e?.detail || "Failed to complete sign-in";
        }
        setError(
          typeof message === "string" ? message : JSON.stringify(message),
        );
      }
    } finally {
      setOauthLoading(false);
      setLoading(false);
      oauthAbortRef.current = null;
    }
  }, [serverUrl, allProviders]);

  // Auto-start OAuth sign-in when server returns 401/403
  useEffect(() => {
    if (authRequired && startOAuthRef.current) {
      startOAuthRef.current = false;
      handleOAuthSignIn();
    }
  }, [authRequired, handleOAuthSignIn]);

  const handleConfirm = useCallback(() => {
    if (!selectedTool) return;

    const availableTools: Record<string, any> = {};
    for (const t of tools) {
      availableTools[t.name] = {
        description: t.description,
        input_schema: t.input_schema,
      };
    }

    onConfirm({
      serverUrl: serverUrl.trim(),
      serverName,
      selectedTool: selectedTool.name,
      toolInputSchema: selectedTool.input_schema,
      availableTools,
      credentials,
    });
    reset();
  }, [
    selectedTool,
    tools,
    serverUrl,
    serverName,
    credentials,
    onConfirm,
    reset,
  ]);

  return (
    <Dialog open={open} onOpenChange={(isOpen) => !isOpen && handleClose()}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>
            {step === "url"
              ? "Connect to MCP Server"
              : `Select a Tool${serverName ? ` — ${serverName}` : ""}`}
          </DialogTitle>
          <DialogDescription>
            {step === "url"
              ? "Enter the URL of an MCP server to discover its available tools."
              : `Found ${tools.length} tool${tools.length !== 1 ? "s" : ""}. Select one to add to your agent.`}
          </DialogDescription>
        </DialogHeader>

        {step === "url" && (
          <div className="flex flex-col gap-4 py-2">
            <div className="flex flex-col gap-2">
              <Label htmlFor="mcp-server-url">Server URL</Label>
              <Input
                id="mcp-server-url"
                type="url"
                placeholder="https://mcp.example.com/mcp"
                value={serverUrl}
                onChange={(e) => setServerUrl(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleDiscoverTools()}
                autoFocus
              />
            </div>

            {/* Auth required: show manual token option */}
            {authRequired && !showManualToken && (
              <button
                onClick={() => setShowManualToken(true)}
                className="text-xs text-gray-500 underline hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
              >
                or enter a token manually
              </button>
            )}

            {/* Manual token entry — only visible when expanded */}
            {showManualToken && (
              <div className="flex flex-col gap-2">
                <Label htmlFor="mcp-auth-token" className="text-sm">
                  Bearer Token
                </Label>
                <Input
                  id="mcp-auth-token"
                  type="password"
                  placeholder="Paste your auth token here"
                  value={manualToken}
                  onChange={(e) => setManualToken(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleDiscoverTools()}
                  autoFocus
                />
              </div>
            )}

            {error && <p className="text-sm text-red-500">{error}</p>}
          </div>
        )}

        {step === "tool" && (
          <ScrollArea className="max-h-[50vh] py-2">
            <div className="flex flex-col gap-2 pr-3">
              {tools.map((tool) => (
                <MCPToolCard
                  key={tool.name}
                  tool={tool}
                  selected={selectedTool?.name === tool.name}
                  onSelect={() => setSelectedTool(tool)}
                />
              ))}
            </div>
          </ScrollArea>
        )}

        <DialogFooter>
          {step === "tool" && (
            <Button
              variant="outline"
              onClick={() => {
                setStep("url");
                setSelectedTool(null);
              }}
            >
              Back
            </Button>
          )}
          <Button variant="outline" onClick={handleClose}>
            Cancel
          </Button>
          {step === "url" && (
            <Button
              onClick={
                authRequired && !showManualToken
                  ? handleOAuthSignIn
                  : handleDiscoverTools
              }
              disabled={!serverUrl.trim() || loading || oauthLoading}
            >
              {loading || oauthLoading ? (
                <span className="flex items-center gap-2">
                  <LoadingSpinner className="size-4" />
                  {oauthLoading ? "Waiting for sign-in..." : "Connecting..."}
                </span>
              ) : authRequired && !showManualToken ? (
                "Sign in & Connect"
              ) : (
                "Discover Tools"
              )}
            </Button>
          )}
          {step === "tool" && (
            <Button onClick={handleConfirm} disabled={!selectedTool}>
              Add Block
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// --------------- Tool Card Component --------------- //

/** Truncate a description to a reasonable length for the collapsed view. */
function truncateDescription(text: string, maxLen = 120): string {
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen).trimEnd() + "…";
}

/** Pretty-print a JSON Schema type for a parameter. */
function schemaTypeLabel(schema: Record<string, any>): string {
  if (schema.type) return schema.type;
  if (schema.anyOf)
    return schema.anyOf.map((s: any) => s.type ?? "any").join(" | ");
  if (schema.oneOf)
    return schema.oneOf.map((s: any) => s.type ?? "any").join(" | ");
  return "any";
}

function MCPToolCard({
  tool,
  selected,
  onSelect,
}: {
  tool: MCPToolResponse;
  selected: boolean;
  onSelect: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const schema = tool.input_schema as Record<string, any>;
  const properties = schema?.properties ?? {};
  const required = new Set<string>(schema?.required ?? []);
  const paramNames = Object.keys(properties);

  // Strip XML-like tags from description for cleaner display.
  // Loop to handle nested tags like <scr<script>ipt> (CodeQL fix).
  let cleanDescription = tool.description ?? "";
  let prev = "";
  while (prev !== cleanDescription) {
    prev = cleanDescription;
    cleanDescription = cleanDescription.replace(/<[^>]*>/g, "");
  }
  cleanDescription = cleanDescription.trim();

  return (
    <button
      onClick={onSelect}
      className={`group flex flex-col rounded-lg border text-left transition-colors ${
        selected
          ? "border-blue-500 bg-blue-50 dark:border-blue-400 dark:bg-blue-950"
          : "border-gray-200 hover:border-gray-300 hover:bg-gray-50 dark:border-slate-700 dark:hover:border-slate-600 dark:hover:bg-slate-800"
      }`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 px-3 pb-1 pt-3">
        <span className="flex-1 text-sm font-semibold dark:text-white">
          {tool.name}
        </span>
        {paramNames.length > 0 && (
          <Badge variant="secondary" className="text-[10px]">
            {paramNames.length} param{paramNames.length !== 1 ? "s" : ""}
          </Badge>
        )}
      </div>

      {/* Description (collapsed: truncated) */}
      {cleanDescription && (
        <p className="px-3 pb-1 text-xs leading-relaxed text-gray-500 dark:text-gray-400">
          {expanded ? cleanDescription : truncateDescription(cleanDescription)}
        </p>
      )}

      {/* Parameter badges (collapsed view) */}
      {!expanded && paramNames.length > 0 && (
        <div className="flex flex-wrap gap-1 px-3 pb-2">
          {paramNames.slice(0, 6).map((name) => (
            <Badge
              key={name}
              variant="outline"
              className="text-[10px] font-normal"
            >
              {name}
              {required.has(name) && (
                <span className="ml-0.5 text-red-400">*</span>
              )}
            </Badge>
          ))}
          {paramNames.length > 6 && (
            <Badge variant="outline" className="text-[10px] font-normal">
              +{paramNames.length - 6} more
            </Badge>
          )}
        </div>
      )}

      {/* Expanded: full parameter details */}
      {expanded && paramNames.length > 0 && (
        <div className="mx-3 mb-2 rounded border border-gray-100 bg-gray-50/50 dark:border-slate-700 dark:bg-slate-800/50">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-gray-100 dark:border-slate-700">
                <th className="px-2 py-1 text-left font-medium text-gray-500 dark:text-gray-400">
                  Parameter
                </th>
                <th className="px-2 py-1 text-left font-medium text-gray-500 dark:text-gray-400">
                  Type
                </th>
                <th className="px-2 py-1 text-left font-medium text-gray-500 dark:text-gray-400">
                  Description
                </th>
              </tr>
            </thead>
            <tbody>
              {paramNames.map((name) => {
                const prop = properties[name] ?? {};
                return (
                  <tr
                    key={name}
                    className="border-b border-gray-50 last:border-0 dark:border-slate-700/50"
                  >
                    <td className="px-2 py-1 font-mono text-[11px] text-gray-700 dark:text-gray-300">
                      {name}
                      {required.has(name) && (
                        <span className="ml-0.5 text-red-400">*</span>
                      )}
                    </td>
                    <td className="px-2 py-1 text-gray-500 dark:text-gray-400">
                      {schemaTypeLabel(prop)}
                    </td>
                    <td className="max-w-[200px] truncate px-2 py-1 text-gray-500 dark:text-gray-400">
                      {prop.description ?? "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Toggle details */}
      {(paramNames.length > 0 || cleanDescription.length > 120) && (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            setExpanded((prev) => !prev);
          }}
          className="flex w-full items-center justify-center gap-1 border-t border-gray-100 py-1.5 text-[10px] text-gray-400 hover:text-gray-600 dark:border-slate-700 dark:text-gray-500 dark:hover:text-gray-300"
        >
          {expanded ? "Hide details" : "Show details"}
          <CaretDown
            className={`h-3 w-3 transition-transform ${expanded ? "rotate-180" : ""}`}
          />
        </button>
      )}
    </button>
  );
}
