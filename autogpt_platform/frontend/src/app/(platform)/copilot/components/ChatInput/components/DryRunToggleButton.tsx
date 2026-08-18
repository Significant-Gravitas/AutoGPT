"use client";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ToggleChip } from "@/components/atoms/ToggleChip/ToggleChip";
import { cn } from "@/lib/utils";
import { FlaskConicalIcon } from "@hugeicons/core-free-icons";

// This button is only rendered on NEW chats (no active session).
// Once a session exists, it is hidden — the session's dry_run flag is
// locked and surfaced via the banner in CopilotPage.tsx instead.

interface Props {
  isDryRun: boolean;
  onToggle: () => void;
  /** "pill" is the pre-brain-dump footer style; "chip" lives in the tray. */
  variant?: "chip" | "pill";
}

export function DryRunToggleButton({
  isDryRun,
  onToggle,
  variant = "chip",
}: Props) {
  const tooltip = isDryRun
    ? "Test mode on — new sessions run without performing real actions (click to turn off)."
    : "Turn on test mode to try prompts without performing real actions.";
  const ariaLabel = isDryRun ? "Test mode active" : "Enable Test mode";

  if (variant === "pill") {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            aria-pressed={isDryRun}
            onClick={onToggle}
            className={cn(
              "inline-flex h-9 items-center justify-center gap-1 rounded-full border border-neutral-200 bg-white px-2.5 text-xs font-medium shadow-sm transition-colors hover:bg-neutral-50",
              isDryRun ? "text-amber-900" : "text-zinc-950 hover:text-zinc-950",
            )}
            aria-label={ariaLabel}
          >
            <Icon icon={FlaskConicalIcon} size={14} />
            <span className="hidden sm:inline">
              {isDryRun ? "Test mode enabled" : "Enable test mode"}
            </span>
          </button>
        </TooltipTrigger>
        <TooltipContent>{tooltip}</TooltipContent>
      </Tooltip>
    );
  }

  return (
    <ToggleChip
      icon={<Icon icon={FlaskConicalIcon} size={14} />}
      label={isDryRun ? "Test mode enabled" : "Enable test mode"}
      tooltip={tooltip}
      ariaLabel={ariaLabel}
      pressed={isDryRun}
      onToggle={onToggle}
      className="sm:min-w-[9.5rem]"
    />
  );
}
