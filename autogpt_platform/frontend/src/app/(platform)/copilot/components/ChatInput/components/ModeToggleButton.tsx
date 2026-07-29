"use client";

import { BrainIcon, LightningIcon, LockIcon } from "@phosphor-icons/react";
import type { CopilotMode } from "../../../store";
import { ToggleChip } from "./ToggleChip";

interface Props {
  mode: CopilotMode;
  onToggle: () => void;
  pinned?: boolean;
}

export function ModeToggleButton({ mode, onToggle, pinned = false }: Props) {
  const isExtended = mode === "extended_thinking";

  const tooltipText = pinned
    ? "Locked to Extended Thinking — building sessions stay on this engine"
    : isExtended
      ? "Extended Thinking — deeper reasoning (click to switch to Fast)"
      : "Fast mode — quicker responses (click to switch to Thinking)";

  const ariaLabel = pinned
    ? "Mode locked to Extended Thinking while building an agent"
    : isExtended
      ? "Switch to Fast mode"
      : "Switch to Extended Thinking mode";

  function getIcon() {
    if (pinned) return <LockIcon size={14} />;
    return isExtended ? <BrainIcon size={14} /> : <LightningIcon size={14} />;
  }

  return (
    <ToggleChip
      icon={getIcon()}
      label={pinned || isExtended ? "Thinking" : "Fast"}
      tooltip={tooltipText}
      ariaLabel={ariaLabel}
      pressed={isExtended}
      onToggle={onToggle}
      locked={pinned}
      className="sm:min-w-[5.5rem]"
    />
  );
}
