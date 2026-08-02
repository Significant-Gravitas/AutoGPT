"use client";

import { FlaskIcon } from "@phosphor-icons/react";
import { ToggleChip } from "./ToggleChip";

// This button is only rendered on NEW chats (no active session).
// Once a session exists, it is hidden — the session's dry_run flag is
// locked and surfaced via the banner in CopilotPage.tsx instead.

interface Props {
  isDryRun: boolean;
  onToggle: () => void;
}

export function DryRunToggleButton({ isDryRun, onToggle }: Props) {
  return (
    <ToggleChip
      icon={<FlaskIcon size={14} />}
      label={isDryRun ? "Test mode enabled" : "Enable test mode"}
      tooltip={
        isDryRun
          ? "Test mode on — new sessions run without performing real actions (click to turn off)."
          : "Turn on test mode to try prompts without performing real actions."
      }
      ariaLabel={isDryRun ? "Test mode active" : "Enable Test mode"}
      pressed={isDryRun}
      onToggle={onToggle}
      className="sm:min-w-[9.5rem]"
    />
  );
}
