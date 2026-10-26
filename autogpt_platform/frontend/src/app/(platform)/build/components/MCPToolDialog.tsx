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
} from "@/components/molecules/Dialog/components/DialogPrimitives";
import { Button } from "@/components/atoms/Button/Button";
import { BaseInput } from "@/components/atoms/Input/BaseInput";
import { Text } from "@/components/atoms/Text/Text";
import { Label } from "@/components/atoms/Label/Label";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Badge } from "@/components/atoms/Badge/Badge";
import { ScrollArea } from "@/components/molecules/ScrollArea/ScrollArea";
import type { CredentialsMetaInput } from "@/lib/autogpt-server-api";
import type { MCPToolResponse } from "@/app/api/__generated__/models/mCPToolResponse";
import {
  postV2DiscoverAvailableToolsOnAnMcpServer,
  postV2StoreABearerTokenForAnMcpServer,
} from "@/app/api/__generated__/endpoints/mcp/mcp";
import { connectMCPOAuth } from "@/lib/mcp-oauth";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
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
import { isKey } from "@/lib/keyboard";
import { mcpServerIdentity, normalizeMcpUrl } from "@/lib/mcp-url";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export type MCPToolDialogResult = {
  serverUrl: string;
  serverName: string | null;
  selectedTool: string;
  toolInputSchema: Record<string, any>;
  availableTools: Record<string, any>;
  /** Credentials meta from the completed authentication flow, null for public servers. */
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
  const [credentialServerUrl, setCredentialServerUrl] = useState<string | null>(
    null,
  );

  const storedAuthScheme: MCPAuthScheme =
    allProviders?.["mcp"]?.savedCredentials.find(
      (credential) =>
        typeof credential.host === "string" &&
        normalizeMcpUrl(credential.host) === normalizeMcpUrl(serverUrl),
    )?.mcp_auth_scheme === "basic"
      ? "basic"
      : "bearer";
  const {
    scheme: manualAuthScheme,
    selectScheme,
    detectSchemeFrom,
    resetScheme,
  } = useMCPAuthScheme(storedAuthScheme, manualToken);

  const startOAuthRef = useRef(false);
  const oauthRequest = useRef<AbortController | null>(null);
  useEffect(() => () => oauthRequest.current?.abort(), []);
  useEffect(() => {
    if (!open) {
      oauthRequest.current?.abort();
      oauthRequest.current = null;
      setOauthLoading(false);
      setLoading(false);
    }
  }, [open]);

  const reset = useCallback(() => {
    oauthRequest.current?.abort();
    oauthRequest.current = null;
    setStep("url");
    setServerUrl("");
    setManualToken("");
    resetScheme();
    setTools([]);
    setServerName(null);
    setLoading(false);
    setError(null);
    setAuthRequired(false);
    setOauthLoading(false);
    setShowManualToken(false);
    setSelectedTool(null);
    setCredentials(null);
    setCredentialServerUrl(null);
  }, [resetScheme]);

  const handleClose = useCallback(() => {
    reset();
    onClose();
  }, [reset, onClose]);

  const applyDiscoveredTools = useCallback(
    (response: {
      data: {
        tools: MCPToolResponse[];
        server_name?: string | null;
      };
    }) => {
      setTools(response.data.tools);
      setServerName(response.data.server_name ?? null);
      setAuthRequired(false);
      setShowManualToken(false);
      setManualToken("");
      resetScheme();
      setStep("tool");
    },
    [resetScheme],
  );

  const discoverTools = useCallback(
    async (url: string) => {
      setLoading(true);
      setError(null);
      try {
        const response = await postV2DiscoverAvailableToolsOnAnMcpServer({
          server_url: url,
          auth_token: null,
        });
        if (response.status !== 200) {
          throw getAPIResponseError(response.status, response.data);
        }
        applyDiscoveredTools(response);
      } catch (error: unknown) {
        const status = getErrorStatus(error);
        if (status === 401 || status === 403) {
          setAuthRequired(true);
          setError(null);
          // Automatically start OAuth sign-in instead of requiring a second click
          setLoading(false);
          startOAuthRef.current = true;
          return;
        }
        setError(getErrorMessage(error, "Failed to connect to MCP server"));
      } finally {
        setLoading(false);
      }
    },
    [applyDiscoveredTools],
  );

  const connectWithManualCredential = useCallback(async () => {
    const url = serverUrl.trim();
    const credential = manualToken.trim();
    if (!url || !credential) return;

    const invalid = validateMCPAuthCredential(credential, manualAuthScheme);
    if (invalid) {
      setError(invalid);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const authValue = prepareMCPAuthCredential(credential, manualAuthScheme);

      // Probe before storing so a rejected credential never replaces a working one.
      const toolsResponse = await postV2DiscoverAvailableToolsOnAnMcpServer({
        server_url: url,
        auth_token: authValue,
      });
      if (toolsResponse.status !== 200) {
        throw getAPIResponseError(toolsResponse.status, toolsResponse.data);
      }

      // Store through the provider so the builder can resolve the new ID.
      const mcpProvider = allProviders?.["mcp"];
      let storedCredential;
      if (mcpProvider) {
        storedCredential = await mcpProvider.mcpStoreToken(url, authValue);
      } else {
        const credentialResponse = await postV2StoreABearerTokenForAnMcpServer({
          server_url: url,
          token: authValue,
        });
        if (credentialResponse.status !== 200) {
          throw getAPIResponseError(
            credentialResponse.status,
            credentialResponse.data,
          );
        }
        storedCredential = credentialResponse.data;
      }

      setCredentials({
        id: storedCredential.id,
        provider: storedCredential.provider,
        type: storedCredential.type,
        title: storedCredential.title,
      });
      setCredentialServerUrl(url);

      applyDiscoveredTools(toolsResponse);
    } catch (error: unknown) {
      setError(
        getErrorMessage(error, "Failed to connect with this credential"),
      );
    } finally {
      setLoading(false);
    }
  }, [
    allProviders,
    applyDiscoveredTools,
    manualAuthScheme,
    manualToken,
    serverUrl,
  ]);

  const handleDiscoverTools = useCallback(() => {
    if (!serverUrl.trim()) return;
    if (showManualToken) {
      void connectWithManualCredential();
      return;
    }
    void discoverTools(serverUrl.trim());
  }, [connectWithManualCredential, discoverTools, serverUrl, showManualToken]);

  const handleOAuthSignIn = useCallback(async () => {
    if (!serverUrl.trim() || oauthRequest.current) return;
    const controller = new AbortController();
    oauthRequest.current = controller;
    const { signal } = controller;
    setError(null);
    setOauthLoading(true);

    try {
      const credential = await connectMCPOAuth({
        serverURL: serverUrl.trim(),
        signal,
        exchange: allProviders?.mcp?.mcpOAuthCallback,
      });
      signal.throwIfAborted();
      if ("reason" in credential) {
        if (credential.noOAuth) setShowManualToken(true);
        setError(credential.reason);
        return;
      }
      setLoading(true);
      setOauthLoading(false);
      setCredentials({
        id: credential.id,
        provider: credential.provider,
        type: credential.type,
        title: credential.title,
      });
      setCredentialServerUrl(serverUrl.trim());
      setAuthRequired(false);
      const response = await postV2DiscoverAvailableToolsOnAnMcpServer(
        { server_url: serverUrl.trim() },
        { signal },
      );
      signal.throwIfAborted();
      if (response.status !== 200) {
        throw getAPIResponseError(response.status, response.data);
      }
      applyDiscoveredTools(response);
    } catch (error: unknown) {
      if (signal.aborted) return;
      const status = getErrorStatus(error);
      const message = getErrorMessage(error, "Failed to complete sign-in");
      if (message === "OAuth flow timed out") {
        setError("OAuth sign-in timed out. Please try again.");
      } else if (status === 401 || status === 403) {
        setError(
          "Authentication succeeded but the server still rejected the request. " +
            "The token audience may not match. Please try again.",
        );
      } else {
        setError(message);
      }
    } finally {
      if (oauthRequest.current === controller) {
        oauthRequest.current = null;
        if (!signal.aborted) {
          setOauthLoading(false);
          setLoading(false);
        }
      }
    }
  }, [serverUrl, allProviders, applyDiscoveredTools]);

  // Auto-start OAuth sign-in when server returns 401/403
  useEffect(() => {
    if (authRequired && startOAuthRef.current) {
      startOAuthRef.current = false;
      void handleOAuthSignIn();
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
      credentials:
        credentialServerUrl === serverUrl.trim() ? credentials : null,
    });
    reset();
  }, [
    selectedTool,
    tools,
    serverUrl,
    serverName,
    credentials,
    credentialServerUrl,
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
              : `Found ${tools.length} tool${tools.length !== 1 ? "s" : ""}. Select one to add to your workflow.`}
          </DialogDescription>
        </DialogHeader>

        {step === "url" && (
          <div className="flex flex-col gap-4 py-2">
            <div className="flex flex-col gap-2">
              <Label htmlFor="mcp-server-url">Server URL</Label>
              <BaseInput
                id="mcp-server-url"
                type="url"
                placeholder="https://mcp.example.com/mcp"
                value={serverUrl}
                onChange={(e) => {
                  const nextUrl = e.target.value;
                  // Only a change of server identity discards the credential.
                  const serverChanged =
                    mcpServerIdentity(serverUrl) !== mcpServerIdentity(nextUrl);
                  setServerUrl(nextUrl);
                  if (!serverChanged) return;

                  if (credentialServerUrl !== nextUrl.trim()) {
                    setCredentials(null);
                    setCredentialServerUrl(null);
                  }
                  setManualToken("");
                  resetScheme();
                  setAuthRequired(false);
                  setShowManualToken(false);
                  setError(null);
                  startOAuthRef.current = false;
                }}
                onKeyDown={(e) => isKey(e, "Enter") && handleDiscoverTools()}
                disabled={loading || oauthLoading}
                autoFocus
              />
            </div>

            {/* Auth required: show manual token option */}
            {authRequired && !showManualToken && (
              <button
                onClick={() => setShowManualToken(true)}
                // TODO(tokens): no exact role: text-gray-500 #6B7280, hover:text-gray-700 #374151; dark-mode classes kept as written (QUALITY 43): no fitting role for dark:text-gray-400 #9CA3AF; #9CA3AF is handle (the builder node handle); no fitting role for dark:hover:text-gray-200 #E5E7EB; #E5E7EB is sidebar-border (a border colour)
                className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                <Text
                  variant="small-tight-underline"
                  as="span"
                  tone="inherit"
                  className="block"
                >
                  or enter an API credential manually
                </Text>
              </button>
            )}

            {/* Manual credential entry — only visible when expanded */}
            {showManualToken && (
              <div className="flex flex-col gap-2">
                <MCPAuthSchemeField
                  value={manualAuthScheme}
                  onChange={selectScheme}
                  disabled={loading || oauthLoading}
                  className="flex flex-col gap-2"
                  labelVariant="body-medium-tight"
                  selectClassName="h-10 rounded-md border border-input bg-background px-3 text-body leading-5"
                />

                <Label htmlFor="mcp-auth-token">
                  <Text variant="body-medium-tight" as="span" tone="inherit">
                    {mcpAuthTokenLabel(manualAuthScheme)}
                  </Text>
                </Label>
                <BaseInput
                  id="mcp-auth-token"
                  aria-describedby="mcp-auth-token-hint"
                  type="password"
                  placeholder={mcpAuthTokenPlaceholder(manualAuthScheme)}
                  value={manualToken}
                  onChange={(e) => {
                    const value = e.target.value;
                    setManualToken(value);
                    detectSchemeFrom(value);
                  }}
                  onKeyDown={(e) => isKey(e, "Enter") && handleDiscoverTools()}
                  disabled={loading || oauthLoading}
                  autoFocus
                />
                <Text
                  variant="small-tight"
                  // TODO(tokens): no exact role: text-gray-500 #6B7280
                  id="mcp-auth-token-hint"
                  className="text-gray-500"
                >
                  {mcpAuthTokenHint(manualAuthScheme)}
                </Text>
              </div>
            )}

            {error && (
              <Text
                variant="body-tight"
                role="alert"
                aria-live="polite"
                // TODO(tokens): dark-mode classes kept as written (QUALITY 43): no exact role: dark:text-red-400 #F26969
                className="text-error-foreground dark:text-red-400"
              >
                {error}
              </Text>
            )}
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
              variant="legacy-outline"
              onClick={() => {
                setStep("url");
                setSelectedTool(null);
              }}
            >
              Back
            </Button>
          )}
          <Button variant="legacy-outline" onClick={handleClose}>
            Cancel
          </Button>
          {step === "url" && (
            <Button
              variant="legacy-default"
              onClick={
                authRequired && !showManualToken
                  ? handleOAuthSignIn
                  : handleDiscoverTools
              }
              disabled={
                !serverUrl.trim() ||
                loading ||
                oauthLoading ||
                (showManualToken && !manualToken.trim())
              }
            >
              {loading || oauthLoading ? (
                <span className="flex items-center gap-2">
                  <LoadingSpinner
                    variant="circle"
                    className="size-4"
                    aria-hidden="true"
                  />
                  {oauthLoading ? "Waiting for sign-in..." : "Connecting..."}
                </span>
              ) : authRequired && !showManualToken ? (
                "Sign in & Connect"
              ) : showManualToken ? (
                "Connect & Discover"
              ) : (
                "Discover Tools"
              )}
            </Button>
          )}
          {step === "tool" && (
            <Button
              variant="legacy-default"
              onClick={handleConfirm}
              disabled={!selectedTool}
            >
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
          ? // TODO(tokens): dark-mode classes kept as written (QUALITY 43): no exact role: dark:border-blue-400 #60A5FA, dark:bg-blue-950 #172554
            // TODO(tokens): no fitting role for border-blue-500 #3B82F6; #3B82F6 is info (a status colour), and this marks a selected card (QUALITY 41(c)); no fitting role for bg-blue-50 #EFF6FF; #EFF6FF is info-surface (a status fill), and this marks a selected card (QUALITY 41(c))
            "border-blue-500 bg-blue-50 dark:border-blue-400 dark:bg-blue-950"
          : // TODO(tokens): no exact role: hover:border-gray-300 #D1D5DB, hover:bg-gray-50 #F9FAFB; no fitting role for border-gray-200 #E5E7EB; #E5E7EB is sidebar-border, whose .dark value (39 39 42) differs (QUALITY 43, SECRT-3154)
            // TODO(tokens): dark-mode classes kept as written (QUALITY 43): no exact role: dark:border-slate-700 #475263, dark:hover:border-slate-600 #5B6A7E, dark:hover:bg-slate-800 #475263
            "border-gray-200 hover:border-gray-300 hover:bg-gray-50 dark:border-slate-700 dark:hover:border-slate-600 dark:hover:bg-slate-800"
      }`}
    >
      {/* Header */}
      <div className="flex items-center gap-2 px-3 pb-1 pt-3">
        <Text
          variant="body-semibold-tight"
          tone="inherit"
          as="span"
          // TODO(tokens): dark-mode classes kept as written (QUALITY 43): no fitting role for dark:text-white #FEFEFE; #FEFEFE is surface (a surface fill), action-foreground (a text colour), node-surface (the builder node card fill)
          className="flex-1 dark:text-white"
        >
          {tool.name}
        </Text>
        {paramNames.length > 0 && (
          <Badge variant="secondary" size="tiny">
            {paramNames.length} param{paramNames.length !== 1 ? "s" : ""}
          </Badge>
        )}
      </div>

      {/* Description (collapsed: truncated) */}
      {cleanDescription && (
        <Text
          variant="small-relaxed"
          // TODO(tokens): no exact role: text-gray-500 #6B7280; dark-mode classes kept as written (QUALITY 43): no fitting role for dark:text-gray-400 #9CA3AF; #9CA3AF is handle (the builder node handle)
          className="px-3 pb-1 text-gray-500 dark:text-gray-400"
        >
          {expanded ? cleanDescription : truncateDescription(cleanDescription)}
        </Text>
      )}

      {/* Parameter badges (collapsed view) */}
      {!expanded && paramNames.length > 0 && (
        <div className="flex flex-wrap gap-1 px-3 pb-2">
          {paramNames.slice(0, 6).map((name) => (
            <Badge key={name} variant="outline" size="tiny-normal">
              {name}
              {required.has(name) && (
                <span
                  // TODO(tokens): no exact role: text-red-400 #F26969
                  className="ml-0.5 text-red-400"
                >
                  *
                </span>
              )}
            </Badge>
          ))}
          {paramNames.length > 6 && (
            <Badge variant="outline" size="tiny-normal">
              +{paramNames.length - 6} more
            </Badge>
          )}
        </div>
      )}

      {/* Expanded: full parameter details */}
      {expanded && paramNames.length > 0 && (
        <div
          // TODO(tokens): no exact role: border-gray-100 #F3F4F6, bg-gray-50/50 #F9FAFB; dark-mode classes kept as written (QUALITY 43): no exact role: dark:border-slate-700 #475263, dark:bg-slate-800/50 #475263
          className="mx-3 mb-2 rounded border border-gray-100 bg-gray-50/50 dark:border-slate-700 dark:bg-slate-800/50"
        >
          <table className="w-full">
            <thead>
              <tr
                // TODO(tokens): no exact role: border-gray-100 #F3F4F6; dark-mode classes kept as written (QUALITY 43): no exact role: dark:border-slate-700 #475263
                className="border-b border-gray-100 dark:border-slate-700"
              >
                <Text
                  variant="small-medium-tight"
                  as="th"
                  // TODO(tokens): no exact role: text-gray-500 #6B7280; dark-mode classes kept as written (QUALITY 43): no fitting role for dark:text-gray-400 #9CA3AF; #9CA3AF is handle (the builder node handle)
                  className="px-2 py-1 text-left text-gray-500 dark:text-gray-400"
                >
                  Parameter
                </Text>
                <Text
                  variant="small-medium-tight"
                  as="th"
                  // TODO(tokens): no exact role: text-gray-500 #6B7280; dark-mode classes kept as written (QUALITY 43): no fitting role for dark:text-gray-400 #9CA3AF; #9CA3AF is handle (the builder node handle)
                  className="px-2 py-1 text-left text-gray-500 dark:text-gray-400"
                >
                  Type
                </Text>
                <Text
                  variant="small-medium-tight"
                  as="th"
                  // TODO(tokens): no exact role: text-gray-500 #6B7280; dark-mode classes kept as written (QUALITY 43): no fitting role for dark:text-gray-400 #9CA3AF; #9CA3AF is handle (the builder node handle)
                  className="px-2 py-1 text-left text-gray-500 dark:text-gray-400"
                >
                  Description
                </Text>
              </tr>
            </thead>
            <tbody>
              {paramNames.map((name) => {
                const prop = properties[name] ?? {};
                return (
                  <tr
                    key={name}
                    // TODO(tokens): no exact role: border-gray-50 #F9FAFB; dark-mode classes kept as written (QUALITY 43): no exact role: dark:border-slate-700/50 #475263
                    className="border-b border-gray-50 last:border-0 dark:border-slate-700/50"
                  >
                    <Text
                      variant="caption-sm-mono"
                      as="td"
                      // TODO(tokens): no exact role: text-gray-700 #374151; dark-mode classes kept as written (QUALITY 43): no exact role: dark:text-gray-300 #D1D5DB
                      className="px-2 py-1 text-gray-700 dark:text-gray-300"
                    >
                      {name}
                      {required.has(name) && (
                        <span
                          // TODO(tokens): no exact role: text-red-400 #F26969
                          className="ml-0.5 text-red-400"
                        >
                          *
                        </span>
                      )}
                    </Text>
                    <Text
                      variant="small-tight"
                      as="td"
                      // TODO(tokens): no exact role: text-gray-500 #6B7280; dark-mode classes kept as written (QUALITY 43): no fitting role for dark:text-gray-400 #9CA3AF; #9CA3AF is handle (the builder node handle)
                      className="px-2 py-1 text-gray-500 dark:text-gray-400"
                    >
                      {schemaTypeLabel(prop)}
                    </Text>
                    <Text
                      variant="small-tight"
                      as="td"
                      // TODO(tokens): no exact role: text-gray-500 #6B7280; dark-mode classes kept as written (QUALITY 43): no fitting role for dark:text-gray-400 #9CA3AF; #9CA3AF is handle (the builder node handle)
                      className="max-w-[200px] truncate px-2 py-1 text-gray-500 dark:text-gray-400"
                    >
                      {prop.description ?? "—"}
                    </Text>
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
          // TODO(tokens): no exact role: border-gray-100 #F3F4F6, hover:text-gray-600 #4B5563; no fitting role for text-gray-400 #9CA3AF; #9CA3AF is handle (the builder node handle); dark-mode classes kept as written (QUALITY 43): no exact role: dark:border-slate-700 #475263, dark:text-gray-500 #6B7280, dark:hover:text-gray-300 #D1D5DB
          className="flex w-full items-center justify-center gap-1 border-t border-gray-100 py-1.5 text-gray-400 hover:text-gray-600 dark:border-slate-700 dark:text-gray-500 dark:hover:text-gray-300"
        >
          <Text variant="tiny-inherit" as="span" tone="inherit">
            {expanded ? "Hide details" : "Show details"}
          </Text>
          <Icon
            icon={ArrowDown01Icon}
            className={`h-3 w-3 transition-transform ${expanded ? "rotate-180" : ""}`}
          />
        </button>
      )}
    </button>
  );
}
