"use client";

import {
  Tooltip,
  TooltipContent,
  TooltipPortal,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { FlaskConicalIcon } from "@hugeicons/core-free-icons";

// This button is only rendered on NEW chats (no active session).
// Once a session exists, it is hidden — the session's dry_run flag is
// locked and surfaced via the banner in CopilotPage.tsx instead.

interface Props {
  isDryRun: boolean;
  onToggle: () => void;
}

export function DryRunToggleButton({ isDryRun, onToggle }: Props) {
  const tooltip = isDryRun
    ? "Test mode on — new sessions run without performing real actions (click to turn off)."
    : "Turn on test mode to try prompts without performing real actions.";
  const ariaLabel = isDryRun ? "Test mode active" : "Enable Test mode";

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-pressed={isDryRun}
          onClick={onToggle}
          className={cn(
            "inline-flex h-8 w-8 items-center justify-center rounded-full transition-colors hover:bg-zinc-100",
            isDryRun
              ? "text-yellow-600 hover:text-yellow-700"
              : "text-muted-foreground hover:text-zinc-700",
          )}
          aria-label={ariaLabel}
        >
          <Icon icon={FlaskConicalIcon} size={16} />
        </button>
      </TooltipTrigger>
      <TooltipPortal>
        <TooltipContent>{tooltip}</TooltipContent>
      </TooltipPortal>
    </Tooltip>
  );
}
