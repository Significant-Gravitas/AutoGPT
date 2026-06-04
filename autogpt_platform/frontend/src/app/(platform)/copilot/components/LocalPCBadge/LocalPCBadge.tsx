"use client";

import { Text } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { Desktop } from "@phosphor-icons/react";

/**
 * Small pill that appears in the copilot UI when the `local-pc-executor`
 * LD flag is on for the current user. Communicates "your turns are
 * routing to your real machine" so the user can correlate platform
 * behavior with what's happening on their disk.
 *
 * Live connection status (machine_id / platform / arch / allowed_root)
 * needs a backend endpoint and is a follow-up. For now the badge is
 * static — its job is to signal "this isn't the cloud sandbox you're
 * used to."
 */
export function LocalPCBadge() {
  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <div className="inline-flex items-center gap-1.5 rounded-full border border-amber-200 bg-amber-50 px-2.5 py-1 text-amber-900">
            <Desktop className="h-3.5 w-3.5" weight="fill" />
            <Text variant="body" className="text-xs font-medium">
              Local PC mode
            </Text>
          </div>
        </TooltipTrigger>
        <TooltipContent side="bottom" sideOffset={6} className="max-w-xs">
          Files and commands route to the{" "}
          <span className="font-mono">autogpt-local-executor</span> shim
          daemon on your machine instead of an E2B cloud sandbox. Make sure{" "}
          <span className="font-mono">autogpt-shim</span> is running. Review
          activity with{" "}
          <span className="font-mono">autogpt-shim audit tail</span>.
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
