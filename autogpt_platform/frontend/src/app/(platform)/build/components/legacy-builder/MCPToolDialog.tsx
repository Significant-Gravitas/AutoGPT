"use client";

import React, { useState, useCallback, useRef, useEffect } from "react";
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
import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import type { MCPTool } from "@/lib/autogpt-server-api";
import { CaretDown } from "@phosphor-icons/react";

export type MCPToolDialogResult = {
  serverUrl: string;
  serverName: string | null;
  selectedTool: string;
  toolInputSchema: Record<string, any>;
  availableTools: Record<string, any>;
};

interface MCPToolDialogProps {
  open: boolean;
  onClose: () => void;
  onConfirm: (result: MCPToolDialogResult) => void;
}

type DialogStep = "url" | "tool";

const OAUTH_TIMEOUT_MS = 5 * 60 * 1000; // 5 minutes
const STORAGE_KEY = "mcp_last_server_url";

export function MCPToolDialog({
  open,
  onClose,
  onConfirm,
}: MCPToolDialogProps) {
  const api = useBackendAPI();

  const [step, setStep] = useState<DialogStep>("url");
  const [serverUrl, setServerUrl] = useState("");
  const [tools, setTools] = useState<MCPTool[]>([]);
  const [serverName, setServerName] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [authRequired, setAuthRequired] = useState(false);
  const [oauthLoading, setOauthLoading] = useState(false);
  const [showManualToken, setShowManualToken] = useState(false);
  const [manualToken, setManualToken] = useState("");
  const [selectedTool, setSelectedTool] = useState<MCPTool | null>(null);

  const oauthLoadingRef = useRef(false);
  const stateTokenRef = useRef<string | null>(null);
  const broadcastChannelRef = useRef<BroadcastChannel | null>(null);
  const messageHandlerRef = useRef<((event: MessageEvent) => void) | null>(
    null,
  );
  const oauthHandledRef = useRef(false);
  const autoConnectAttemptedRef = useRef(false);

  // Pre-fill last used server URL when dialog opens (without auto-connecting)
  useEffect(() => {
    if (!open) {
      autoConnectAttemptedRef.current = false;
      return;
    }

    if (autoConnectAttemptedRef.current) return;
    autoConnectAttemptedRef.current = true;

    const lastUrl = localStorage.getItem(STORAGE_KEY);
    if (lastUrl) {
      setServerUrl(lastUrl);
    }
  }, [open]);

  // Clean up listeners on unmount
  useEffect(() => {
    return () => {
      if (messageHandlerRef.current) {
        window.removeEventListener("message", messageHandlerRef.current);
      }
      if (broadcastChannelRef.current) {
        broadcastChannelRef.current.close();
      }
    };
  }, []);

  const cleanupOAuthListeners = useCallback(() => {
    if (messageHandlerRef.current) {
      window.removeEventListener("message", messageHandlerRef.current);
      messageHandlerRef.current = null;
    }
    if (broadcastChannelRef.current) {
      broadcastChannelRef.current.close();
      broadcastChannelRef.current = null;
    }
    setOauthLoading(false);
    oauthLoadingRef.current = false;
    oauthHandledRef.current = false;
  }, []);

  const reset = useCallback(() => {
    cleanupOAuthListeners();
    setStep("url");
    setServerUrl("");
    setManualToken("");
    setTools([]);
    setServerName(null);
    setLoading(false);
    setError(null);
    setAuthRequired(false);
    setShowManualToken(false);
    setSelectedTool(null);
    stateTokenRef.current = null;
  }, [cleanupOAuthListeners]);

  const handleClose = useCallback(() => {
    reset();
    onClose();
  }, [reset, onClose]);

  const discoverTools = useCallback(
    async (url: string, authToken?: string) => {
      setLoading(true);
      setError(null);
      try {
        const result = await api.mcpDiscoverTools(url, authToken);
        localStorage.setItem(STORAGE_KEY, url);
        setTools(result.tools);
        setServerName(result.server_name);
        setAuthRequired(false);
        setShowManualToken(false);
        setStep("tool");
      } catch (e: any) {
        if (e?.status === 401 || e?.status === 403) {
          setAuthRequired(true);
          setError(null);
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
    },
    [api],
  );

  const handleDiscoverTools = useCallback(() => {
    if (!serverUrl.trim()) return;
    discoverTools(serverUrl.trim(), manualToken.trim() || undefined);
  }, [serverUrl, manualToken, discoverTools]);

  const handleOAuthResult = useCallback(
    async (data: {
      success: boolean;
      code?: string;
      state?: string;
      message?: string;
    }) => {
      // Prevent double-handling (BroadcastChannel + postMessage may both fire)
      if (oauthHandledRef.current) return;
      oauthHandledRef.current = true;

      if (!data.success) {
        setError(data.message || "OAuth authentication failed.");
        cleanupOAuthListeners();
        return;
      }

      cleanupOAuthListeners();
      setAuthRequired(false);

      // Exchange code for tokens (stored server-side)
      setLoading(true);
      try {
        await api.mcpOAuthCallback(data.code!, stateTokenRef.current!);
        // Retry discovery — backend auto-uses stored credential
        const result = await api.mcpDiscoverTools(serverUrl.trim());
        localStorage.setItem(STORAGE_KEY, serverUrl.trim());
        setTools(result.tools);
        setServerName(result.server_name);
        setStep("tool");
      } catch (e: any) {
        const message = e?.message || e?.detail || "Failed to complete sign-in";
        setError(
          typeof message === "string" ? message : JSON.stringify(message),
        );
      } finally {
        setLoading(false);
      }
    },
    [api, serverUrl, cleanupOAuthListeners],
  );

  const handleOAuthSignIn = useCallback(async () => {
    if (!serverUrl.trim()) return;
    setError(null);
    oauthHandledRef.current = false;

    // Open popup SYNCHRONOUSLY (before async call) to avoid browser popup blockers
    const width = 500;
    const height = 700;
    const left = window.screenX + (window.outerWidth - width) / 2;
    const top = window.screenY + (window.outerHeight - height) / 2;
    const popup = window.open(
      "about:blank",
      "mcp_oauth",
      `width=${width},height=${height},left=${left},top=${top},scrollbars=yes`,
    );

    setOauthLoading(true);
    oauthLoadingRef.current = true;

    try {
      const { login_url, state_token } = await api.mcpOAuthLogin(
        serverUrl.trim(),
      );
      stateTokenRef.current = state_token;

      if (popup && !popup.closed) {
        popup.location.href = login_url;
      } else {
        // Popup was blocked — open in new tab as fallback
        window.open(login_url, "_blank");
      }

      // Listener 1: BroadcastChannel (works even when window.opener is null)
      const bc = new BroadcastChannel("mcp_oauth");
      bc.onmessage = (event) => {
        if (event.data?.type === "mcp_oauth_result") {
          handleOAuthResult(event.data);
        }
      };
      broadcastChannelRef.current = bc;

      // Listener 2: window.postMessage (fallback)
      const handleMessage = (event: MessageEvent) => {
        if (event.origin !== window.location.origin) return;
        if (event.data?.message_type === "mcp_oauth_result") {
          handleOAuthResult(event.data);
        }
      };
      messageHandlerRef.current = handleMessage;
      window.addEventListener("message", handleMessage);

      // Timeout
      setTimeout(() => {
        if (oauthLoadingRef.current) {
          cleanupOAuthListeners();
          setError("OAuth sign-in timed out. Please try again.");
        }
      }, OAUTH_TIMEOUT_MS);
    } catch (e: any) {
      if (popup && !popup.closed) popup.close();

      // If server doesn't support OAuth → show manual token entry
      if (e?.status === 400) {
        setShowManualToken(true);
        setError(
          "This server does not support OAuth sign-in. Please enter a token manually.",
        );
      } else {
        const message = e?.message || "Failed to initiate sign-in";
        setError(
          typeof message === "string" ? message : JSON.stringify(message),
        );
      }
      cleanupOAuthListeners();
    }
  }, [api, serverUrl, handleOAuthResult, cleanupOAuthListeners]);

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
    });
    reset();
  }, [selectedTool, tools, serverUrl, onConfirm, reset]);

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

            {/* Auth required: show sign-in panel */}
            {authRequired && (
              <div className="flex flex-col items-center gap-3 rounded-lg border border-amber-200 bg-amber-50 p-4 dark:border-amber-800 dark:bg-amber-950">
                <p className="text-sm font-medium text-amber-700 dark:text-amber-300">
                  This server requires authentication
                </p>
                <Button
                  onClick={handleOAuthSignIn}
                  disabled={oauthLoading || loading}
                  className="w-full"
                >
                  {oauthLoading ? (
                    <span className="flex items-center gap-2">
                      <LoadingSpinner className="size-4" />
                      Waiting for sign-in...
                    </span>
                  ) : (
                    "Sign in"
                  )}
                </Button>
                {!showManualToken && (
                  <button
                    onClick={() => setShowManualToken(true)}
                    className="text-xs text-gray-500 underline hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
                  >
                    or enter a token manually
                  </button>
                )}
              </div>
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
              onClick={handleDiscoverTools}
              disabled={!serverUrl.trim() || loading || oauthLoading}
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <LoadingSpinner className="size-4" />
                  Connecting...
                </span>
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
  tool: MCPTool;
  selected: boolean;
  onSelect: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const properties = tool.input_schema?.properties ?? {};
  const required = new Set<string>(tool.input_schema?.required ?? []);
  const paramNames = Object.keys(properties);

  // Strip XML-like tags and hints from description for cleaner display
  const cleanDescription = (tool.description ?? "")
    .replace(/<[^>]+>[^<]*<\/[^>]+>/g, "")
    .replace(/<[^>]+>/g, "")
    .trim();

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
