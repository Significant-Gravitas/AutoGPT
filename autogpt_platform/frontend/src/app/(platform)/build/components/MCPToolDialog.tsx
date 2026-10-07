"use client";

import React, {
  useState,
  useCallback,
  useRef,
  useEffect,
  useContext,
} from "react";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Badge } from "@/components/atoms/Badge/Badge";
import { ScrollArea } from "@/components/__legacy__/ui/scroll-area";
import { cn } from "@/lib/utils";
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
    <Dialog
      title={
        step === "url"
          ? "Connect to MCP Server"
          : `Select a Tool${serverName ? ` — ${serverName}` : ""}`
      }
      controlled={{
        isOpen: open,
        set: (isOpen) => {
          if (!isOpen) handleClose();
        },
      }}
      styling={{ maxWidth: "32rem", minWidth: "32rem" }}
    >
      <Dialog.Content>
        <Text variant="body" tone="secondary">
          {step === "url"
            ? "Enter the URL of an MCP server to discover its available tools."
            : `Found ${tools.length} tool${tools.length !== 1 ? "s" : ""}. Select one to add to your agent.`}
        </Text>

        {step === "url" && (
          <div className="flex flex-col gap-4 py-2">
            <Input
              id="mcp-server-url"
              label="Server URL"
              labelVariant="body-medium"
              size="md"
              wrapperClassName="mb-0"
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

            {/* Auth required: show manual token option */}
            {authRequired && !showManualToken && (
              <Button
                variant="link"
                onClick={() => setShowManualToken(true)}
                className="h-auto min-w-0 p-0 text-xs font-normal text-muted-foreground hover:text-zinc-700"
              >
                or enter an API credential manually
              </Button>
            )}

            {/* Manual credential entry — only visible when expanded */}
            {showManualToken && (
              <div className="flex flex-col gap-2">
                <MCPAuthSchemeField
                  value={manualAuthScheme}
                  onChange={selectScheme}
                  disabled={loading || oauthLoading}
                  className="flex flex-col gap-2"
                  labelClassName="text-sm font-medium"
                  selectClassName="h-10 rounded-md border border-input bg-background px-3 text-sm"
                />

                <Input
                  id="mcp-auth-token"
                  label={mcpAuthTokenLabel(manualAuthScheme)}
                  labelVariant="body"
                  size="md"
                  wrapperClassName="mb-0"
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
                <Text id="mcp-auth-token-hint" variant="small" tone="muted">
                  {mcpAuthTokenHint(manualAuthScheme)}
                </Text>
              </div>
            )}

            {error && (
              <Text
                variant="body"
                role="alert"
                aria-live="polite"
                className="text-red-700"
                unmask={false}
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

        <Dialog.Footer className="gap-2">
          {step === "tool" && (
            <Button
              variant="outline"
              size="md"
              onClick={() => {
                setStep("url");
                setSelectedTool(null);
              }}
            >
              Back
            </Button>
          )}
          <Button variant="outline" size="md" onClick={handleClose}>
            Cancel
          </Button>
          {step === "url" && (
            <Button
              size="md"
              loading={loading || oauthLoading}
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
              {loading || oauthLoading
                ? oauthLoading
                  ? "Waiting for sign-in..."
                  : "Connecting..."
                : authRequired && !showManualToken
                  ? "Sign in & Connect"
                  : showManualToken
                    ? "Connect & Discover"
                    : "Discover Tools"}
            </Button>
          )}
          {step === "tool" && (
            <Button size="md" onClick={handleConfirm} disabled={!selectedTool}>
              Add Block
            </Button>
          )}
        </Dialog.Footer>
      </Dialog.Content>
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
      className={cn(
        "group flex flex-col rounded-lg border text-left transition-colors",
        selected
          ? "border-blue-500 bg-blue-50"
          : "border-zinc-200 hover:border-zinc-300 hover:bg-zinc-50",
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-2 px-3 pt-3 pb-1">
        <Text
          variant="body-medium"
          as="span"
          className="flex-1 font-semibold"
          unmask={false}
        >
          {tool.name}
        </Text>
        {paramNames.length > 0 && (
          <Badge variant="info" size="small">
            {paramNames.length} param{paramNames.length !== 1 ? "s" : ""}
          </Badge>
        )}
      </div>

      {/* Description (collapsed: truncated) */}
      {cleanDescription && (
        <Text
          variant="small"
          tone="muted"
          className="px-3 pb-1 leading-relaxed"
          unmask={false}
        >
          {expanded ? cleanDescription : truncateDescription(cleanDescription)}
        </Text>
      )}

      {/* Parameter badges (collapsed view) */}
      {!expanded && paramNames.length > 0 && (
        <div className="flex flex-wrap gap-1 px-3 pb-2">
          {paramNames.slice(0, 6).map((name) => (
            <Badge key={name} variant="info" size="small">
              {name}
              {required.has(name) && (
                <span className="-ml-1 text-red-400">*</span>
              )}
            </Badge>
          ))}
          {paramNames.length > 6 && (
            <Badge variant="info" size="small">
              +{paramNames.length - 6} more
            </Badge>
          )}
        </div>
      )}

      {/* Expanded: full parameter details */}
      {expanded && paramNames.length > 0 && (
        <div className="mx-3 mb-2 rounded-sm border border-zinc-100 bg-zinc-50/50">
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-zinc-100">
                <th className="px-2 py-1 text-left font-medium text-muted-foreground">
                  Parameter
                </th>
                <th className="px-2 py-1 text-left font-medium text-muted-foreground">
                  Type
                </th>
                <th className="px-2 py-1 text-left font-medium text-muted-foreground">
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
                    className="border-b border-zinc-50 last:border-0"
                  >
                    <td className="px-2 py-1 font-mono text-[11px] text-zinc-700">
                      {name}
                      {required.has(name) && (
                        <span className="ml-0.5 text-red-400">*</span>
                      )}
                    </td>
                    <td className="px-2 py-1 text-muted-foreground">
                      {schemaTypeLabel(prop)}
                    </td>
                    <td className="max-w-[200px] truncate px-2 py-1 text-muted-foreground">
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
        <Button
          variant="ghost"
          size="sm"
          onClick={(e) => {
            e.stopPropagation();
            setExpanded((prev) => !prev);
          }}
          className="w-full gap-1 rounded-none border-0 border-t border-zinc-100 py-1.5 text-[10px] font-normal text-zinc-400 hover:border-zinc-100 hover:bg-transparent hover:text-zinc-600"
        >
          {expanded ? "Hide details" : "Show details"}
          <Icon
            icon={ArrowDown01Icon}
            className={cn(
              "h-3 w-3 transition-transform",
              expanded && "rotate-180",
            )}
          />
        </Button>
      )}
    </button>
  );
}
