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
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import {
  CaretDownIcon,
  RobotIcon,
  TerminalIcon,
  WarningCircleIcon,
} from "@phosphor-icons/react";
import { useContext, useEffect } from "react";
import { getSavedCodexCredentials } from "../../../helpers/copilotLlmAuth";
import { useCopilotUIStore } from "../../../store";

export function LlmRouteSelector() {
  const isCodexEnabled = useGetFlag(Flag.CODEX_SUBSCRIPTION_COPILOT);
  const providers = useContext(CredentialsProvidersContext);
  const codexCredentials = getSavedCodexCredentials(providers);
  const { copilotLlmAuth, setCopilotLlmAuth } = useCopilotUIStore();
  const selectedCredential =
    copilotLlmAuth.authProvider === "codex"
      ? codexCredentials.find(
          (credential) => credential.id === copilotLlmAuth.credentialId,
        )
      : null;

  useEffect(() => {
    if (copilotLlmAuth.authProvider !== "codex") return;
    if (isCodexEnabled && (providers === null || selectedCredential)) return;

    toast({
      variant: "destructive",
      title: "ChatGPT/Codex connection unavailable",
      description: isCodexEnabled
        ? "The selected connection was removed. Choose a connection before starting a new task."
        : "ChatGPT/Codex is not available for new AutoPilot tasks right now.",
    });
  }, [copilotLlmAuth, isCodexEnabled, providers, selectedCredential]);

  const isCodexSelected = copilotLlmAuth.authProvider === "codex";
  const isSelectionUnavailable =
    isCodexSelected &&
    (!isCodexEnabled || (providers !== null && !selectedCredential));
  if (!isCodexSelected && (!isCodexEnabled || codexCredentials.length === 0)) {
    return null;
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          aria-label={`AI connection: ${isSelectionUnavailable ? "ChatGPT/Codex unavailable" : isCodexSelected ? "ChatGPT/Codex" : "AutoGPT platform"} — change connection`}
          className={cn(
            "ml-2 inline-flex h-9 items-center gap-1.5 rounded-full border border-neutral-200 bg-white px-2.5 text-xs font-medium shadow-sm transition-colors hover:bg-neutral-50",
            isSelectionUnavailable
              ? "text-red-600"
              : isCodexSelected
                ? "text-emerald-600"
                : "text-zinc-700",
          )}
        >
          {isSelectionUnavailable ? (
            <WarningCircleIcon size={14} />
          ) : isCodexSelected ? (
            <TerminalIcon size={14} />
          ) : (
            <RobotIcon size={14} />
          )}
          <span className="hidden sm:inline">
            {isSelectionUnavailable
              ? "Connection missing"
              : isCodexSelected
                ? "ChatGPT/Codex"
                : "AutoGPT"}
          </span>
          <CaretDownIcon className="size-3 text-zinc-400" weight="bold" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-64">
        <DropdownMenuItem
          onClick={() =>
            setCopilotLlmAuth({
              authProvider: "platform",
              credentialId: null,
            })
          }
          className={cn("gap-2", !isCodexSelected && "bg-zinc-100")}
        >
          <RobotIcon size={16} />
          <div>
            <div>AutoGPT platform</div>
            <div className="text-xs text-zinc-500">Uses platform credits</div>
          </div>
        </DropdownMenuItem>
        {isSelectionUnavailable && (
          <DropdownMenuItem disabled className="gap-2 text-red-600">
            <WarningCircleIcon size={16} />
            Selected ChatGPT/Codex connection is unavailable
          </DropdownMenuItem>
        )}
        {isCodexEnabled &&
          codexCredentials.map((credential) => (
            <DropdownMenuItem
              key={credential.id}
              onClick={() =>
                setCopilotLlmAuth({
                  authProvider: "codex",
                  credentialId: credential.id,
                })
              }
              className={cn(
                "gap-2",
                selectedCredential?.id === credential.id && "bg-zinc-100",
              )}
            >
              <TerminalIcon size={16} />
              <div className="min-w-0">
                <div>ChatGPT/Codex</div>
                <div className="truncate text-xs text-zinc-500">
                  {credential.title ??
                    credential.username ??
                    "Connected account"}
                  {" · Uses your ChatGPT plan"}
                </div>
              </div>
            </DropdownMenuItem>
          ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
