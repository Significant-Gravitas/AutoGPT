"use client";

import { useGetV2ListChatTransports } from "@/app/api/__generated__/endpoints/chat/chat";
import type { ChatTransportResponse } from "@/app/api/__generated__/models/chatTransportResponse";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { toast } from "@/components/molecules/Toast/use-toast";
import { cn } from "@/lib/utils";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import Link from "next/link";
import { useContext, useEffect } from "react";
import {
  PiCaretDown as CaretDownIcon,
  PiKey as KeyIcon,
  PiWarningCircle as WarningCircleIcon,
} from "react-icons/pi";
import {
  findSelectedLLMTransport,
  getAvailableLLMTransports,
  getChatTransportSelection,
  resolveCopilotLLMAuthSelection,
} from "../../../helpers/copilotLlmAuth";
import { useCopilotUIStore } from "../../../store";

function getTransportDescription(
  transport: ChatTransportResponse,
  credentialTitle: string | undefined,
): string {
  if (transport.auth_provider === "codex") {
    return credentialTitle
      ? `${credentialTitle} · Uses your ChatGPT plan`
      : "Uses your ChatGPT plan";
  }
  return transport.label === "AutoGPT Platform"
    ? "Uses your AutoGPT plan"
    : "Uses this server's configured chat provider";
}

export function LLMRouteSelector() {
  const providers = useContext(CredentialsProvidersContext);
  const transportQuery = useGetV2ListChatTransports({
    query: { refetchOnWindowFocus: true, staleTime: 0 },
  });
  const transports =
    transportQuery.data?.status === 200
      ? transportQuery.data.data.transports
      : undefined;
  const availableTransports = getAvailableLLMTransports(transports);
  const {
    copilotLlmAuth: copilotLLMAuth,
    setCopilotLlmAuth: setCopilotLLMAuth,
  } = useCopilotUIStore();
  const resolvedSelection = resolveCopilotLLMAuthSelection(
    transports,
    copilotLLMAuth,
  );
  const selectedTransport = findSelectedLLMTransport(
    availableTransports,
    copilotLLMAuth,
  );
  const selectedConnectionMissing =
    transports !== undefined &&
    availableTransports.length > 0 &&
    copilotLLMAuth.authProvider !== "platform" &&
    !selectedTransport;

  useEffect(() => {
    if (!resolvedSelection) return;
    if (
      resolvedSelection.authProvider === copilotLLMAuth.authProvider &&
      resolvedSelection.credentialId === copilotLLMAuth.credentialId
    ) {
      return;
    }

    setCopilotLLMAuth(resolvedSelection);
  }, [
    copilotLLMAuth.authProvider,
    copilotLLMAuth.credentialId,
    resolvedSelection?.authProvider,
    resolvedSelection?.credentialId,
    setCopilotLLMAuth,
  ]);

  useEffect(() => {
    if (!selectedConnectionMissing) return;

    toast({
      title: "AI connections changed",
      description:
        "The selected connection is unavailable. The next AutoPilot task will use the available default.",
    });
  }, [selectedConnectionMissing]);

  if (transportQuery.isPending && transports === undefined) return null;

  if (transports === undefined) {
    return (
      <span
        aria-label="AI connections unavailable"
        className="ml-2 inline-flex h-9 items-center gap-1.5 rounded-full border border-destructive/20 bg-destructive/10 px-2.5 text-xs font-medium text-destructive"
      >
        <WarningCircleIcon size={14} />
        <span className="hidden sm:inline">Connections unavailable</span>
      </span>
    );
  }

  if (availableTransports.length === 0) {
    return (
      <Link
        href="/settings/integrations"
        aria-label="Set up an AI connection"
        className="ml-2 inline-flex h-9 items-center gap-1.5 rounded-full border border-border bg-muted px-2.5 text-xs font-medium text-foreground transition-colors hover:bg-muted/80"
      >
        <WarningCircleIcon size={14} />
        <span className="hidden sm:inline">Set up AI connection</span>
      </Link>
    );
  }

  if (availableTransports.length === 1) return null;

  const selectionLabel = selectedConnectionMissing
    ? "Connection unavailable"
    : (selectedTransport?.label ?? "Choose connection");

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          aria-label={`AI connection: ${selectionLabel} — change connection`}
          className={cn(
            "ml-2 inline-flex h-9 items-center gap-1.5 rounded-full border border-border bg-background px-2.5 text-xs font-medium shadow-sm transition-colors hover:bg-muted",
            selectedConnectionMissing ? "text-destructive" : "text-foreground",
          )}
        >
          {selectedConnectionMissing ? (
            <WarningCircleIcon size={14} />
          ) : (
            <KeyIcon size={14} />
          )}
          <span className="hidden sm:inline">{selectionLabel}</span>
          <CaretDownIcon size={12} className="text-muted-foreground" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-72">
        {selectedConnectionMissing && (
          <DropdownMenuItem disabled className="gap-2 text-destructive">
            <WarningCircleIcon size={16} />
            Selected connection is unavailable
          </DropdownMenuItem>
        )}
        {availableTransports.map((transport) => {
          const selection = getChatTransportSelection(transport);
          if (!selection) return null;
          const credential =
            transport.auth_provider === "codex"
              ? providers?.codex?.savedCredentials.find(
                  (candidate) => candidate.id === transport.credential_id,
                )
              : undefined;
          const isSelected =
            selectedTransport?.auth_provider === transport.auth_provider &&
            selectedTransport.credential_id === transport.credential_id;
          return (
            <DropdownMenuItem
              key={`${transport.auth_provider}:${transport.credential_id ?? "deployment"}`}
              onClick={() => setCopilotLLMAuth(selection)}
              className={cn("gap-2", isSelected && "bg-muted")}
            >
              <KeyIcon size={16} />
              <div className="min-w-0">
                <div>{transport.label}</div>
                <div className="truncate text-xs text-muted-foreground">
                  {getTransportDescription(transport, credential?.title)}
                </div>
              </div>
            </DropdownMenuItem>
          );
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
