"use client";

import { Text } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { Desktop } from "@phosphor-icons/react";
import { parseAsString, useQueryState } from "nuqs";
import { useLocalPCExecutor } from "../../hooks/useLocalPCExecutor";

const PLATFORM_DISPLAY: Record<string, string> = {
  darwin: "macOS",
  linux: "Linux",
  windows: "Windows",
  wsl2: "Windows (WSL2)",
};

/**
 * Pill in the copilot UI when the `local-pc-executor` LD flag is on.
 *
 * When the backend reports a connected shim for the active session, the
 * pill shows the user's actual machine ("Local PC: macOS arm64
 * ~/autogpt-workspace"). When no shim is connected, falls back to the
 * generic "Local PC mode" label so the user knows the *mode* is on
 * even if their daemon isn't running yet.
 */
export function LocalPCBadge() {
  const [sessionId] = useQueryState("sessionId", parseAsString);
  const { data: executor } = useLocalPCExecutor(sessionId);

  const connected = executor?.kind === "shim";
  const platformLabel = executor?.platform
    ? (PLATFORM_DISPLAY[executor.platform] ?? executor.platform)
    : null;

  const label = connected
    ? `Local PC: ${platformLabel}${executor?.arch ? ` ${executor.arch}` : ""}`
    : "Local PC mode";

  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <div
            className={
              connected
                ? "inline-flex items-center gap-1.5 rounded-full border border-green-200 bg-green-50 px-2.5 py-1 text-green-900"
                : "inline-flex items-center gap-1.5 rounded-full border border-amber-200 bg-amber-50 px-2.5 py-1 text-amber-900"
            }
          >
            <Desktop className="h-3.5 w-3.5" weight="fill" />
            <Text variant="body" className="text-xs font-medium">
              {label}
            </Text>
          </div>
        </TooltipTrigger>
        <TooltipContent side="bottom" sideOffset={6} className="max-w-xs">
          {connected ? (
            <div className="flex flex-col gap-1">
              <div>
                Files and commands route to{" "}
                <span className="font-mono">autogpt-local-executor</span> on
                your machine.
              </div>
              {executor?.allowed_root ? (
                <div>
                  Workspace:{" "}
                  <span className="font-mono">{executor.allowed_root}</span>
                </div>
              ) : null}
              {executor?.machine_id ? (
                <div className="text-[10px] opacity-70">
                  machine {executor.machine_id.slice(0, 12)}
                </div>
              ) : null}
              {executor?.computer_use_features &&
              executor.computer_use_features.length > 0 ? (
                <div className="text-[10px] opacity-70">
                  computer-use: {executor.computer_use_features.join(", ")}
                </div>
              ) : null}
              <div className="mt-1 text-[10px] opacity-70">
                Review with{" "}
                <span className="font-mono">autogpt-shim audit tail</span>
              </div>
            </div>
          ) : (
            <>
              Local PC mode is enabled but no shim is currently connected
              for this session. Start{" "}
              <span className="font-mono">autogpt-shim start</span> on your
              machine, or wait for it to reconnect.
            </>
          )}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
