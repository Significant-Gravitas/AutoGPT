"use client";

import { Text } from "@/components/atoms/Text/Text";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { DesktopIcon } from "@phosphor-icons/react";
import { useLocalPCExecutor } from "../../hooks/useLocalPCExecutor";

const PLATFORM_DISPLAY: Record<string, string> = {
  darwin: "macOS",
  linux: "Linux",
  windows: "Windows",
  wsl2: "Windows (WSL2)",
};

interface Props {
  sessionID: string | null;
  machineID?: string | null;
  allowedRoot?: string | null;
}

export function LocalPCBadge({ sessionID, machineID, allowedRoot }: Props) {
  const { data: executor, isError, isLoading } = useLocalPCExecutor(sessionID);

  const connected = executor?.kind === "shim";
  const platformLabel = executor?.platform
    ? (PLATFORM_DISPLAY[executor.platform] ?? executor.platform)
    : "unknown platform";
  const label = isLoading
    ? "Checking Local PC…"
    : isError
      ? "Local PC status unavailable"
      : connected
        ? `Local PC connected: ${platformLabel}${executor?.arch ? ` ${executor.arch}` : ""}`
        : sessionID
          ? "Local PC disconnected"
          : "Local PC";

  const consentLabel =
    executor?.computer_use_consent === "approved"
      ? "Computer control allowed for this session"
      : executor?.computer_use_consent === "denied"
        ? "Computer control denied for this session"
        : executor?.computer_use_consent === "pending"
          ? "Computer control awaiting your decision"
          : null;

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label={`${label}. Open Local PC details`}
          aria-live="polite"
          className={
            connected
              ? "inline-flex min-h-11 max-w-full items-center gap-1.5 rounded-full border border-green-200 bg-green-50 px-2.5 py-1 text-left text-green-900 outline-none focus-visible:ring-2 focus-visible:ring-green-700 focus-visible:ring-offset-2"
              : "inline-flex min-h-11 max-w-full items-center gap-1.5 rounded-full border border-amber-200 bg-amber-50 px-2.5 py-1 text-left text-amber-900 outline-none focus-visible:ring-2 focus-visible:ring-amber-700 focus-visible:ring-offset-2"
          }
        >
          <DesktopIcon
            className="h-3.5 w-3.5 shrink-0"
            weight="fill"
            aria-hidden="true"
          />
          <Text
            variant="body"
            className="min-w-0 break-words text-xs font-medium"
          >
            {label}
          </Text>
        </button>
      </PopoverTrigger>
      <PopoverContent
        side="bottom"
        sideOffset={6}
        align="start"
        className="w-[min(24rem,calc(100vw-2rem))] motion-reduce:animate-none"
      >
        <div className="flex min-w-0 flex-col gap-2 text-sm">
          {connected ? (
            <>
              <div>
                Files and commands route to{" "}
                <span className="font-mono">autogpt-local-executor</span> on
                your machine.
              </div>
              {executor?.capabilities?.includes("shell") ? (
                <div className="text-xs text-neutral-600">
                  Shell commands are not limited by that file root and run with
                  your full user-level permissions.
                </div>
              ) : null}
              {executor?.machine_id ? (
                <div className="break-all text-xs text-neutral-600">
                  machine {executor.machine_id.slice(0, 12)}
                </div>
              ) : null}
              {(executor?.computer_use_features_coarse ?? []).length > 0 ? (
                <div className="break-words text-xs text-neutral-600">
                  computer-use:{" "}
                  {executor?.computer_use_features_coarse?.join(", ")}
                </div>
              ) : (executor?.computer_use_features ?? []).length > 0 ? (
                <div className="break-words text-xs text-neutral-600">
                  computer-use: {executor?.computer_use_features?.join(", ")}
                </div>
              ) : null}
              {consentLabel ? (
                <div className="text-xs text-neutral-600">{consentLabel}</div>
              ) : null}
              <div className="text-xs text-neutral-600">
                Review with{" "}
                <span className="font-mono">autogpt-shim audit tail</span>
              </div>
            </>
          ) : (
            <div>
              {isLoading
                ? "Checking whether your local executor is connected."
                : isError
                  ? "The local executor status could not be loaded. It will retry automatically."
                  : "The Local PC selected for this chat is offline. Restart autogpt-shim on that computer to reconnect."}
            </div>
          )}

          {executor?.allowed_root || allowedRoot ? (
            <div>
              File root (<span className="font-mono">FILE_*</span> only):{" "}
              <span className="break-all font-mono">
                {executor?.allowed_root ?? allowedRoot}
              </span>
            </div>
          ) : null}

          {!connected && machineID ? (
            <div className="break-all text-xs text-neutral-600">
              machine {machineID.slice(0, 12)}
            </div>
          ) : null}
        </div>
      </PopoverContent>
    </Popover>
  );
}
