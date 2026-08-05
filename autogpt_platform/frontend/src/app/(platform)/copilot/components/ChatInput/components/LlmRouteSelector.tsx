"use client";

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
import { Icon } from "@/components/atoms/Icon/Icon";
import { useContext, useEffect } from "react";
import {
  getConnectedSubsidizedLlmTransports,
  getSubsidizedTransportSelection,
  resolveCopilotLlmAuthSelection,
} from "../../../helpers/copilotLlmAuth";
import { useCopilotUIStore } from "../../../store";

export function LlmRouteSelector() {
  const providers = useContext(CredentialsProvidersContext);
  const transports = getConnectedSubsidizedLlmTransports(providers);
  const { copilotLlmAuth, setCopilotLlmAuth } = useCopilotUIStore();
  const resolvedSelection = resolveCopilotLlmAuthSelection(
    providers,
    copilotLlmAuth,
  );
  const selectedTransport = transports.find(
    (transport) => transport.authProvider === copilotLlmAuth.authProvider,
  );
  const selectedCredential = selectedTransport?.credentials.find(
    (credential) => credential.id === copilotLlmAuth.credentialId,
  );
  const selectedConnectionMissing =
    providers !== null &&
    copilotLlmAuth.authProvider !== "platform" &&
    !selectedCredential;

  useEffect(() => {
    if (transports.length > 1 || !resolvedSelection) return;
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
    transports.length,
  ]);

  useEffect(() => {
    if (!selectedConnectionMissing || transports.length === 1) return;

    toast({
      variant: transports.length === 0 ? "default" : "destructive",
      title:
        transports.length === 0
          ? "AI connections changed"
          : "ChatGPT/Codex connection unavailable",
      description:
        transports.length === 0
          ? "The next AutoPilot task will resolve the currently available connection before it starts."
          : "The selected connection was removed. Choose another connected subscription before starting a new task.",
    });
  }, [selectedConnectionMissing, transports.length]);

  if (transports.length <= 1) return null;

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
      <DropdownMenuContent align="start" className="w-64">
        {selectedConnectionMissing && (
          <DropdownMenuItem disabled className="gap-2 text-red-600">
            <Icon icon={Alert01Icon} size={16} />
            Selected connection is unavailable
          </DropdownMenuItem>
        )}
        {transports.map((transport) => {
          const selection = getSubsidizedTransportSelection(
            transport,
            copilotLlmAuth,
          );
          const credential = transport.credentials.find(
            (candidate) => candidate.id === selection.credentialId,
          );
          return (
            <DropdownMenuItem
              key={transport.authProvider}
              onClick={() => setCopilotLlmAuth(selection)}
              className={cn(
                "gap-2",
                selectedTransport?.authProvider === transport.authProvider &&
                  selectedCredential &&
                  "bg-zinc-100",
              )}
            >
              <Icon icon={Key01Icon} size={16} />
              <div className="min-w-0">
                <div>{transport.label}</div>
                <div className="truncate text-xs text-zinc-500">
                  {credential?.title ??
                    credential?.username ??
                    "Connected account"}
                  {` · ${transport.description}`}
                </div>
              </div>
            </DropdownMenuItem>
          );
        })}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
