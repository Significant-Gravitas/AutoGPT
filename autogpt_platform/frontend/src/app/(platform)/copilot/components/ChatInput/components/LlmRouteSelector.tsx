"use client";

import { useGetV2ListChatTransports } from "@/app/api/__generated__/endpoints/chat/chat";
import type { ChatTransportResponse } from "@/app/api/__generated__/models/chatTransportResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { toast } from "@/components/molecules/Toast/use-toast";
import { cn } from "@/lib/utils";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import {
  Alert01Icon,
  ArrowDown01Icon,
  Key01Icon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import { useContext, useEffect } from "react";
import {
  findSelectedLlmTransport,
  getAvailableLlmTransports,
  getChatTransportSelection,
  resolveCopilotLlmAuthSelection,
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

export function LlmRouteSelector() {
  const providers = useContext(CredentialsProvidersContext);
  const transportQuery = useGetV2ListChatTransports({
    query: { refetchOnWindowFocus: true, staleTime: 0 },
  });
  const transports =
    transportQuery.data?.status === 200
      ? transportQuery.data.data.transports
      : undefined;
  const availableTransports = getAvailableLlmTransports(transports);
  const { copilotLlmAuth, setCopilotLlmAuth } = useCopilotUIStore();
  const resolvedSelection = resolveCopilotLlmAuthSelection(
    transports,
    copilotLlmAuth,
  );
  const selectedTransport = findSelectedLlmTransport(
    availableTransports,
    copilotLlmAuth,
  );
  const selectedConnectionMissing =
    transports !== undefined &&
    availableTransports.length > 0 &&
    copilotLlmAuth.authProvider !== "platform" &&
    !selectedTransport;

  useEffect(() => {
    if (!resolvedSelection) return;
    if (
      resolvedSelection.authProvider === copilotLlmAuth.authProvider &&
      resolvedSelection.credentialId === copilotLlmAuth.credentialId
    ) {
      return;
    }

    setCopilotLlmAuth(resolvedSelection);
  }, [
    copilotLlmAuth.authProvider,
    copilotLlmAuth.credentialId,
    resolvedSelection?.authProvider,
    resolvedSelection?.credentialId,
    setCopilotLlmAuth,
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
        className="ml-2 inline-flex h-9 items-center gap-1.5 rounded-full border border-red-200 bg-red-50 px-2.5 text-xs font-medium text-red-600"
      >
        <Icon icon={Alert01Icon} size={14} />
        <span className="hidden sm:inline">Connections unavailable</span>
      </span>
    );
  }

  if (availableTransports.length === 0) {
    return (
      <Link
        href="/settings/integrations"
        aria-label="Set up an AI connection"
        className="ml-2 inline-flex h-9 items-center gap-1.5 rounded-full border border-amber-200 bg-amber-50 px-2.5 text-xs font-medium text-amber-700 transition-colors hover:bg-amber-100"
      >
        <Icon icon={Alert01Icon} size={14} />
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
            "ml-2 inline-flex h-9 items-center gap-1.5 rounded-full border border-neutral-200 bg-white px-2.5 text-xs font-medium shadow-sm transition-colors hover:bg-neutral-50",
            selectedConnectionMissing ? "text-red-600" : "text-emerald-600",
          )}
        >
          {selectedConnectionMissing ? (
            <Icon icon={Alert01Icon} size={14} />
          ) : (
            <Icon icon={Key01Icon} size={14} />
          )}
          <span className="hidden sm:inline">{selectionLabel}</span>
          <Icon icon={ArrowDown01Icon} size={12} className="text-zinc-400" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-72">
        {selectedConnectionMissing && (
          <DropdownMenuItem disabled className="gap-2 text-red-600">
            <Icon icon={Alert01Icon} size={16} />
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
              onClick={() => setCopilotLlmAuth(selection)}
              className={cn("gap-2", isSelected && "bg-zinc-100")}
            >
              <Icon icon={Key01Icon} size={16} />
              <div className="min-w-0">
                <div>{transport.label}</div>
                <div className="truncate text-xs text-zinc-500">
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
